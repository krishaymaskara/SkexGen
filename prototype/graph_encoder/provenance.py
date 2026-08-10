"""Strict C6 source, configuration, and partition provenance.

The Graph V1 implementation established the stable behavior: collect exact Git
identity and a content digest, reject dirty or changed source, and recheck
immediately before checkpoint publication.  C6 permits either a named branch
or a detached checkout because authoritative Adroit validation uses detached
exact commits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

from prototype.flat_baseline.provenance import source_tree_sha256

from .config import CHECKPOINT_SCHEMA
from .decoder_contract import POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
from .errors import GraphEncoderError
from .partitions import (
    ASSIGNMENT_SHA256,
    AUTHORITATIVE_FILE_SHA256,
    FAMILY_COUNTS,
    SPLIT_NAME,
    TRAIN_PARTITION,
)


C6_PROVENANCE_VERSION = "GE1-C6-PROVENANCE-v1"
TRAINING_PARTITION_IDENTITY_VERSION = "GE1-TRAIN-PARTITION-v1"


@dataclass(frozen=True)
class C6Provenance:
    version: str
    git_branch: object
    detached_head: bool
    git_commit: str
    git_dirty: bool
    git_status_porcelain: tuple
    source_tree_sha256: str
    configuration_sha256: str
    partition_identity_sha256: str
    python_version: str
    pytorch_version: str
    device: str
    host: str
    slurm_job_id: object
    slurm_array_task_id: object
    seed: int
    encoder_arm: str
    checkpoint_schema: str
    operation_magnitude_parameterization: str = (
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    )

    def to_dict(self):
        values = asdict(self)
        values["git_status_porcelain"] = list(self.git_status_porcelain)
        return values


@dataclass(frozen=True)
class C6ProvenanceContext:
    repository_root: str
    expected_commit: str
    expected_source_digest: str
    expected_configuration_digest: str
    expected_partition_digest: str
    seed: int
    encoder_arm: str
    device: str
    authorized: C6Provenance
    expected_operation_magnitude_parameterization: str = (
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    )


def sha256_json(value):
    """Hash one JSON-compatible value with deterministic UTF-8 encoding."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_partition_identity(family_ids):
    """Return payload-free identity for one authorized train selection."""

    ordered = tuple(sorted(family_ids))
    if (
        not ordered
        or len(ordered) != len(set(ordered))
        or any(not isinstance(item, str) or not item for item in ordered)
    ):
        raise GraphEncoderError(
            "invalid_training_partition_identity",
            "training family IDs must be nonempty, unique strings",
        )
    assignment_hashes = dict(ASSIGNMENT_SHA256)
    family_counts = dict(FAMILY_COUNTS)
    selected_payload = ("\n".join(ordered) + "\n").encode("utf-8")
    selected_hash = hashlib.sha256(selected_payload).hexdigest()
    if len(ordered) not in (4, 32, family_counts[TRAIN_PARTITION]):
        raise GraphEncoderError(
            "invalid_training_partition_identity",
            "C6 permits only the 4/32-family gates or complete train assignment",
        )
    if (
        len(ordered) == family_counts[TRAIN_PARTITION]
        and selected_hash != assignment_hashes[TRAIN_PARTITION]
    ):
        raise GraphEncoderError(
            "partition_identity_mismatch",
            "complete train family assignment hash differs from authority",
        )
    return {
        "version": TRAINING_PARTITION_IDENTITY_VERSION,
        "manifest_name": SPLIT_NAME,
        "manifest_sha256": AUTHORITATIVE_FILE_SHA256,
        "partition": TRAIN_PARTITION,
        "authoritative_partition_assignment_sha256": assignment_hashes[
            TRAIN_PARTITION
        ],
        "authoritative_partition_family_count": family_counts[TRAIN_PARTITION],
        "selected_family_count": len(ordered),
        "selected_family_ids_sha256": selected_hash,
        "protected_payload_content_present": False,
    }


def configuration_digest(model_config, training_config, partition_identity):
    """Bind frozen model/training configuration and train partition identity."""

    return sha256_json({
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "partition": partition_identity,
    })


def collect_c6_provenance(
    repository_root,
    *,
    configuration_sha256,
    partition_identity_sha256,
    seed,
    encoder_arm,
    device,
    operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
    torch_module=None
):
    """Collect the exact C6 run identity without reading corpus payloads."""

    root = _validated_repository_root(repository_root)
    commit = _git(root, ("rev-parse", "--verify", "HEAD"), "missing_git_commit")
    branch_result = _git_optional(
        root, ("symbolic-ref", "--quiet", "--short", "HEAD")
    )
    branch = branch_result if branch_result else None
    status_output = _git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "malformed_git_status",
        allow_empty=True,
    )
    status = tuple(line for line in status_output.splitlines() if line)
    pytorch_version = "unavailable"
    if torch_module is not None:
        pytorch_version = str(torch_module.__version__)
    return C6Provenance(
        C6_PROVENANCE_VERSION,
        branch,
        branch is None,
        commit,
        bool(status),
        status,
        source_tree_sha256(root),
        configuration_sha256,
        partition_identity_sha256,
        platform.python_version(),
        pytorch_version,
        str(device),
        socket.gethostname(),
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
        int(seed),
        str(encoder_arm),
        CHECKPOINT_SCHEMA,
        str(operation_magnitude_parameterization),
    )


def authorize_c6_provenance(
    repository_root,
    *,
    expected_commit,
    configuration_sha256,
    partition_identity_sha256,
    seed,
    encoder_arm,
    device,
    operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
    torch_module=None
):
    """Collect and freeze the clean provenance expected throughout one run."""

    snapshot = collect_c6_provenance(
        repository_root,
        configuration_sha256=configuration_sha256,
        partition_identity_sha256=partition_identity_sha256,
        seed=seed,
        encoder_arm=encoder_arm,
        device=device,
        operation_magnitude_parameterization=(
            operation_magnitude_parameterization
        ),
        torch_module=torch_module,
    )
    _validate_snapshot(
        snapshot,
        expected_commit=expected_commit,
        expected_source_digest=snapshot.source_tree_sha256,
        expected_configuration_digest=configuration_sha256,
        expected_partition_digest=partition_identity_sha256,
        seed=seed,
        encoder_arm=encoder_arm,
        expected_operation_magnitude_parameterization=(
            operation_magnitude_parameterization
        ),
    )
    return C6ProvenanceContext(
        str(Path(repository_root).resolve()),
        expected_commit,
        snapshot.source_tree_sha256,
        configuration_sha256,
        partition_identity_sha256,
        int(seed),
        str(encoder_arm),
        str(device),
        snapshot,
        str(operation_magnitude_parameterization),
    )


def verify_c6_provenance(
    context,
    torch_module=None,
    configuration_sha256=None,
    partition_identity_sha256=None,
):
    """Recollect and require exact run identity before checkpoint writing."""

    if not isinstance(context, C6ProvenanceContext):
        raise GraphEncoderError(
            "invalid_provenance_context", "context must be C6ProvenanceContext"
        )
    current_configuration = (
        context.expected_configuration_digest
        if configuration_sha256 is None else configuration_sha256
    )
    current_partition = (
        context.expected_partition_digest
        if partition_identity_sha256 is None else partition_identity_sha256
    )
    actual = collect_c6_provenance(
        context.repository_root,
        configuration_sha256=current_configuration,
        partition_identity_sha256=current_partition,
        seed=context.seed,
        encoder_arm=context.encoder_arm,
        device=context.device,
        operation_magnitude_parameterization=(
            context.expected_operation_magnitude_parameterization
        ),
        torch_module=torch_module,
    )
    _validate_snapshot(
        actual,
        expected_commit=context.expected_commit,
        expected_source_digest=context.expected_source_digest,
        expected_configuration_digest=context.expected_configuration_digest,
        expected_partition_digest=context.expected_partition_digest,
        seed=context.seed,
        encoder_arm=context.encoder_arm,
        expected_operation_magnitude_parameterization=(
            context.expected_operation_magnitude_parameterization
        ),
    )
    if actual.to_dict() != context.authorized.to_dict():
        # Host/job/runtime fields are stable inside one run. A change is
        # terminal because it means the checkpoint no longer describes the
        # authorized process.
        raise GraphEncoderError(
            "provenance_record_changed",
            "current provenance differs from the authorized run record",
        )
    return actual


def _validate_snapshot(
    snapshot,
    *,
    expected_commit,
    expected_source_digest,
    expected_configuration_digest,
    expected_partition_digest,
    seed,
    encoder_arm,
    expected_operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
):
    if not isinstance(snapshot, C6Provenance):
        raise GraphEncoderError(
            "invalid_provenance_record", "snapshot must be C6Provenance"
        )
    if snapshot.version != C6_PROVENANCE_VERSION:
        _fail("provenance_identity_mismatch", "provenance version differs")
    if snapshot.git_commit != expected_commit:
        _fail("wrong_git_commit", "checked-out commit differs")
    if snapshot.git_dirty or snapshot.git_status_porcelain:
        _fail("dirty_source_tree", "tracked or untracked source is dirty")
    if snapshot.source_tree_sha256 != expected_source_digest:
        _fail("source_tree_digest_mismatch", "governed source digest differs")
    if snapshot.configuration_sha256 != expected_configuration_digest:
        _fail("configuration_digest_mismatch", "configuration digest differs")
    if snapshot.partition_identity_sha256 != expected_partition_digest:
        _fail("partition_identity_mismatch", "manifest/partition identity differs")
    if snapshot.seed != seed or snapshot.encoder_arm != encoder_arm:
        _fail("run_identity_mismatch", "seed or encoder arm differs")
    if snapshot.checkpoint_schema != CHECKPOINT_SCHEMA:
        _fail("checkpoint_identity_mismatch", "checkpoint schema differs")
    if (
        snapshot.operation_magnitude_parameterization
        != expected_operation_magnitude_parameterization
    ):
        _fail(
            "operation_magnitude_parameterization_mismatch",
            "operation-magnitude parameterization differs",
        )
    if snapshot.detached_head is not (snapshot.git_branch is None):
        _fail("malformed_git_identity", "branch/detached state is inconsistent")
    return snapshot


def _validated_repository_root(repository_root):
    if repository_root is None:
        _fail("missing_repository_root", "repository_root must be explicit")
    root = Path(repository_root).resolve()
    actual = Path(_git(
        root, ("rev-parse", "--show-toplevel"), "missing_repository_root"
    )).resolve()
    if actual != root:
        _fail("wrong_repository_root", "repository_root is not the Git root")
    return root


def _git(root, arguments, error_code, allow_empty=False):
    try:
        result = subprocess.run(
            ("git",) + tuple(arguments),
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError as exc:
        raise GraphEncoderError(error_code, str(exc)) from exc
    value = result.stdout.rstrip("\r\n") if allow_empty else result.stdout.strip()
    if result.returncode != 0 or (not allow_empty and not value):
        _fail(error_code, "Git provenance command failed")
    return value


def _git_optional(root, arguments):
    try:
        result = subprocess.run(
            ("git",) + tuple(arguments),
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _fail(code, detail):
    raise GraphEncoderError(code, detail)
