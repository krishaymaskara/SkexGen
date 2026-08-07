"""GE1 configuration, guarded data access, batching, and C4 encoders."""

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
    )
