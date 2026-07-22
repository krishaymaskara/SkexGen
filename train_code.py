# This entry point trains the autoregressive prior over the ten discrete codes
# extracted from the already-trained sketch and extrusion encoders.
import os
import torch
import argparse
from model.code import CodeModel
from dataset import CodeDataset
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter



def train(args):
    # Restrict PyTorch to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Load unique ten-code rows from code.pkl and shuffle them into mini-batches.
    dataset = CodeDataset(datapath=args.input, maxlen=args.seqlen) 
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=5)
    # Build an eight-layer Transformer that predicts each next code in sequence.
    model = CodeModel(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=args.seqlen,
        classes=args.code,
    )
    model = model.to(device).train()
    
    # A single Adam optimizer updates every parameter in the code model.
    network_parameters = list(model.parameters()) 
    optimizer = torch.optim.Adam(network_parameters, lr=1e-3)
   
    # TensorBoard stores the training-loss history in --output.
    writer = SummaryWriter(log_dir=args.output)
    
    # Revisit the code dataset 800 times.
    iters = 0
    print('Start training...')

    for epoch in range(800): 
        print(epoch)

        for batch in dataloader:
            # Move one batch of ten-code sequences to the GPU.
            code = batch
            code = code.to(device)

            # Teacher-force with all but the final token and predict the full
            # shifted sequence returned by the model.
            logits = model(code[:, :-1])

            # Flatten sequence positions and calculate categorical next-code loss.
            c_pred = logits.reshape(-1, logits.shape[-1]) 
            c_target = code.reshape(-1)
            code_loss = F.cross_entropy(c_pred, c_target)
           
            total_loss = code_loss

            # Record the loss every 20 updates.
            if iters % 20 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters)

            # Backpropagate, clip large gradients, and update the Transformer.
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(network_parameters, max_norm=1.0)  # clip gradient
            optimizer.step()
            iters += 1

        writer.flush()

        # Save the learned code prior every 500 epochs.
        if (epoch+1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(args.output,'code_epoch_'+str(epoch+1)+'.pt'))

    writer.close()


if __name__ == "__main__":
    # Read the extracted-code file, output path, and training dimensions.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--code", type=int, required=True)
    args = parser.parse_args()

    # Create the log/checkpoint folder if needed, then start training.
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    train(args)
