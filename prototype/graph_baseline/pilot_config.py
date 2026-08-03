"""Protected partition boundary for the graph-v1 IID pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass


GRAPH_PILOT_IDENTITY = (
    "B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1-iid-pilot-v1"
)
EXPECTED_TRAIN_EXAMPLES = 544
EXPECTED_VALIDATION_EXAMPLES = 68
EXPECTED_OPTIMIZER_STEPS = 136
EXPECTED_EXAMPLES_PROCESSED = 1088


@dataclass(frozen=True)
class GraphPilotConfig:
    pilot_identity: str = GRAPH_PILOT_IDENTITY
    seed: int = 2026
    epochs: int = 2
    batch_size: int = 8
    device: str = "cpu"
    split_name: str = "iid"
    training_partition: str = "train"
    validation_partition: str = "validation"
    validation_partition_identity: str = "iid_validation"
    checkpoint_selection_metric: str = "ordinary_validation_total_loss"
    output_dir: str = "constrained_graph_v1_position_bias_c1_runs"
    systematic_partition_accessed: bool = False
    test_partition_accessed: bool = False
    require_clean_source: bool = True

    def validate(self):
        fixed = {
            "pilot_identity": GRAPH_PILOT_IDENTITY,
            "seed": 2026,
            "epochs": 2,
            "batch_size": 8,
            "device": "cpu",
            "split_name": "iid",
            "training_partition": "train",
            "validation_partition": "validation",
            "validation_partition_identity": "iid_validation",
            "checkpoint_selection_metric": "ordinary_validation_total_loss",
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError("{} differs from frozen graph pilot".format(name))
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be nonempty")
        if type(self.require_clean_source) is not bool:
            raise ValueError("require_clean_source must be Boolean")

    def to_dict(self):
        self.validate()
        return asdict(self)


def validate_partition_authorization(config):
    config.validate()
    if config.systematic_partition_accessed is not False or config.test_partition_accessed is not False:
        raise ValueError("protected partition access is forbidden")
