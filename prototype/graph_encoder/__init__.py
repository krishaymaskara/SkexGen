"""GE1 configuration, guarded data access, encoders, and C5 decoder."""

from .batching import (
    FlatEncoderInput,
    GraphBookkeeping,
    GraphNodeContent,
    GraphSemanticInput,
    PairedBatch,
    build_paired_batch,
    permute_graph,
)

from .canonicalization import (
    CanonicalizedGraph,
    GraphCanonicalizationInput,
    canonicalize_graph,
)
from .config import (
    GE1Config,
    GE1TrainingConfig,
    frozen_encoder_config,
    frozen_feedforward_width,
)
from .errors import GraphEncoderError
from .partitions import load_development, load_train
from .decoder_contract import (
    AUTONOMOUS_OUTPUT_VERSION,
    BOOKKEEPING_ONLY_VALUES,
    COMMON_LOSS_VERSION,
    OUTPUT_POSITION_CONTRACT_VERSION,
    OUTPUT_POSITION_SIGNALS,
    SHARED_DECODER_VERSION,
    output_position_contract_metadata,
)

__all__ = (
    "GE1Config",
    "GE1TrainingConfig",
    "GraphEncoderError",
    "FlatEncoderInput",
    "GraphBookkeeping",
    "CanonicalizedGraph",
    "GraphCanonicalizationInput",
    "GraphNodeContent",
    "GraphSemanticInput",
    "PairedBatch",
    "build_paired_batch",
    "canonicalize_graph",
    "frozen_encoder_config",
    "frozen_feedforward_width",
    "load_development",
    "load_train",
    "permute_graph",
    "AUTONOMOUS_OUTPUT_VERSION",
    "BOOKKEEPING_ONLY_VALUES",
    "COMMON_LOSS_VERSION",
    "OUTPUT_POSITION_CONTRACT_VERSION",
    "OUTPUT_POSITION_SIGNALS",
    "SHARED_DECODER_VERSION",
    "output_position_contract_metadata",
)

try:
    from .encoders import (
        EncodedMemory,
        FlatProgramEncoder,
        TypedGraphProgramEncoder,
        capacity_difference_percent,
        default_encoder_config,
        encoder_parameter_report,
        shared_initialization_source,
    )
    from .checkpoint import (
        GE1CheckpointError,
        ge1_checkpoint_payload,
        load_ge1_checkpoint,
        save_ge1_checkpoint,
    )
    from .losses import GE1Loss, common_ge1_loss
    from .model import (
        GE1Model,
        GE1TeacherForcedOutput,
        build_ge1_model,
        build_matched_ge1_models,
        canonical_shared_decoder,
    )
    from .shared_decoder import (
        AutonomousRawRow,
        DecoderParityError,
        ExplicitConversionOutcome,
        SharedDecoderPrediction,
        SharedGE1Decoder,
        assert_exact_tensor_parity,
    )
except ImportError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ = __all__ + (
        "EncodedMemory",
        "FlatProgramEncoder",
        "TypedGraphProgramEncoder",
        "capacity_difference_percent",
        "default_encoder_config",
        "encoder_parameter_report",
        "shared_initialization_source",
        "AutonomousRawRow",
        "DecoderParityError",
        "ExplicitConversionOutcome",
        "GE1CheckpointError",
        "GE1Loss",
        "GE1Model",
        "GE1TeacherForcedOutput",
        "SharedDecoderPrediction",
        "SharedGE1Decoder",
        "assert_exact_tensor_parity",
        "build_ge1_model",
        "build_matched_ge1_models",
        "canonical_shared_decoder",
        "common_ge1_loss",
        "ge1_checkpoint_payload",
        "load_ge1_checkpoint",
        "save_ge1_checkpoint",
    )
