# This is the complete unconditional-generation entry point. It samples ten
# discrete codes, decodes sketch/extrusion tokens, parses them as CAD histories,
# and writes the project's intermediate OBJ-style representation.
import os
import torch
import argparse
from multiprocessing import Pool
from model.code import CodeModel
from model.decoder import SketchDecoder, EXTDecoder
from model.encoder import PARAMEncoder, CMDEncoder, EXTEncoder

import sys

# Load CAD parsing and serialization helpers from the local utility package.
sys.path.insert(0, 'utils')
from utils import CADparser, write_obj_sample

# Generation size, code-model batch size, and CPU parser-worker count are fixed
# constants in the original script rather than command-line arguments.
NUM_TRHEADS = 36 
NUM_SAMPLE = 20000
BS = 1024

def sample(args):
    # Expose the requested physical GPU as cuda:0 inside this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")
  
    # Recreate and load the four-code sketch-topology encoder. Sampling uses its
    # learned codebook embeddings to turn sampled indices back into latent vectors.
    cmd_encoder = CMDEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=200,
        code_len = 4,
        num_code = 500,
    )
    cmd_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'cmdenc_epoch_300.pt')))
    cmd_encoder = cmd_encoder.to(device).eval()

    # Recreate and load the two-code sketch-geometry encoder.
    param_encoder = PARAMEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=args.bit,
        max_len=200,
        code_len = 2,
        num_code = 1000,
    )
    param_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'paramenc_epoch_300.pt')))
    param_encoder = param_encoder.to(device).eval()

    # Load the autoregressive sketch decoder trained with those six sketch codes.
    sketch_decoder = SketchDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        pix_len=200,
        cmd_len=124,
        quantization_bits=args.bit,
    )
    sketch_decoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'sketchdec_epoch_300.pt')))
    sketch_decoder = sketch_decoder.to(device).eval()
    
    # Load the four-code extrusion encoder to access its learned codebook.
    ext_encoder = EXTEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        quantization_bits=args.bit,
        max_len=96,
        code_len = 4,
        num_code = 1000,
    )
    ext_encoder.load_state_dict(torch.load(os.path.join(args.ext_weight, 'extenc_epoch_200.pt')))
    ext_encoder = ext_encoder.to(device).eval()

    # Load the autoregressive decoder that reconstructs extrusion sequences.
    ext_decoder = EXTDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        max_len=96,
        quantization_bits=args.bit,
    )
    ext_decoder.load_state_dict(torch.load(os.path.join(args.ext_weight, 'extdec_epoch_200.pt')))
    ext_decoder = ext_decoder.to(device).eval()

    # Load the code prior that generates complete ten-code combinations.
    code_model = CodeModel(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=10,
        classes=1000,
    )
    code_model.load_state_dict(torch.load(os.path.join(args.code_weight, 'code_epoch_800.pt')))
    code_model = code_model.to(device).eval()

    print('Random Generation...')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Hold decoded CAD token histories and direct references to the three learned
    # embedding tables used to look up sampled code indices.
    cad = []
    cmd_codebook = cmd_encoder.vq_vae._embedding
    param_codebook = param_encoder.vq_vae._embedding
    ext_codebook = ext_encoder.vq_vae._embedding
  
    # Keep sampling batches until at least 20,000 decoded histories exist.
    while len(cad) < NUM_SAMPLE:
        with torch.no_grad():
            # Sample ten indices and split them into topology, sketch geometry,
            # and extrusion groups using the fixed 4+2+4 layout.
            codes = code_model.sample(n_samples=BS)
            cmd_code = codes[:,:4] 
            param_code = codes[:,4:6] 
            ext_code = codes[:,6:] 

            # Reject rows whose command indices exceed the smaller 500-entry
            # topology codebook; the other two codebooks both have 1,000 entries.
            cmd_codes = []
            param_codes = []
            ext_codes = []
            for cmd, param, ext in zip(cmd_code, param_code, ext_code):
                if torch.max(cmd) >= 500:
                    continue
                else:
                    cmd_codes.append(cmd)
                    param_codes.append(param)
                    ext_codes.append(ext)
            cmd_codes = torch.vstack(cmd_codes)
            param_codes = torch.vstack(param_codes)
            ext_codes = torch.vstack(ext_codes)

            # Convert valid indices to codebook embeddings and project them into
            # the latent spaces expected by the decoders.
            latent_cmd = cmd_encoder.up(cmd_codebook(cmd_codes))
            latent_param = param_encoder.up(param_codebook(param_codes))
            latent_ext = ext_encoder.up(ext_codebook(ext_codes))
            latent_sketch = torch.cat((latent_cmd, latent_param), 1)
                
        # Autoregressively decode sketch tokens. The sketch decoder also produces
        # per-sketch extrusion latents that align later operations with sketches.
        sample_pixels, latent_ext_samples = sketch_decoder.sample(n_samples=latent_sketch.shape[0], \
                        latent_z=latent_sketch, latent_ext=latent_ext)
        _latent_ext_ = torch.vstack(latent_ext_samples)

        # Decode extrusion tokens conditioned on both extrusion codes and sketches.
        sample_merges = ext_decoder.sample(n_samples=len(sample_pixels), latent_z=_latent_ext_, sample_pixels=sample_pixels)
        cad += sample_merges
        print(f'cad:{len(cad)}')
        
    # Parse decoded token histories in parallel on CPU. Invalid histories are
    # silently discarded by raster_cad.
    gen_data = []

    load_iter = Pool(NUM_TRHEADS).imap(raster_cad, cad) 
    for data_sample in load_iter:
        gen_data += data_sample
    print(len(gen_data))

    # Save each valid parsed history in its own zero-padded output directory.
    print('Saving...')
    print('Writting OBJ...')
    for index, value in enumerate(gen_data):
        output = os.path.join(args.output, str(index).zfill(5))
        if not os.path.exists(output):
            os.makedirs(output)
        write_obj_sample(output, value)


def raster_cad(pixels):   
    # Convert one generated token history into structured CAD data. Returning an
    # empty list lets the multiprocessing aggregation skip parse failures.
    try:
        parser = CADparser(args.bit)
        parsed_data = parser.perform(pixels)
        return [parsed_data]
    except Exception as error_msg:  
        return []


if __name__ == "__main__":
    # Parse output/checkpoint paths, GPU selection, and coordinate bit depth.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sketch_weight", type=str, required=True)
    parser.add_argument("--ext_weight", type=str, required=True)
    parser.add_argument("--code_weight", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bit", type=int, required=True)
    args = parser.parse_args()
    
    sample(args)
