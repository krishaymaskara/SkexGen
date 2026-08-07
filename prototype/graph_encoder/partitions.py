"""Operation-template-only data access for GE1 C1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from prototype.model_data.loader import load_partition_physical_examples

from .errors import GraphEncoderError


SPLIT_NAME = "operation_template"
TRAIN_PARTITION = "train"
DEVELOPMENT_PARTITION = "validation"
LOADABLE_PARTITIONS = (TRAIN_PARTITION, DEVELOPMENT_PARTITION)
# RR is opened once at Stage 7 and ER stays closed throughout GE1. Neither may
# ever be reached by relaxing this loader. A future RR evaluation must add a
# separate, explicitly named, separately audited entry point so that protected
# access is visible in the call graph and reviewable on its own.
PROTECTED_PARTITIONS = ("secondary_systematic_validation", "test")

AUTHORITATIVE_RELATIVE_FILE = "manifests/operation_template.json"
AUTHORITATIVE_ADROIT_PATH = (
    "/scratch/network/km6349/controlled_corpora/"
    "b0-pilot-680-seed2026/manifests/operation_template.json"
)
AUTHORITATIVE_FILE_SHA256 = (
    "a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7"
)

FAMILY_COUNTS = (
    ("secondary_systematic_validation", 114),
    ("test", 114),
    ("train", 407),
    ("validation", 45),
)
SAMPLE_COUNTS = (
    ("secondary_systematic_validation", 228),
    ("test", 228),
    ("train", 814),
    ("validation", 90),
)
ASSIGNMENT_SHA256 = (
    (
        "secondary_systematic_validation",
        "eb37468f4baf6540891add9293a66aee2077ecd7cb64d7cb486902a6eb58c494",
    ),
    ("test", "b18df0bdf9cc95575cc79f6cd2bdded5f9559e0964305a525969a2702de22663"),
    ("train", "42d61d2224ae1a2110279f66d374516f8b5f220271407913147d8be390c66595"),
    (
        "validation",
        "9af48e34e0ae2e5fd266d2336b6adecdad543c75a2c470ea132b9c7a593afaf6",
    ),
)
PARTITION_TEMPLATES = (
    ("secondary_systematic_validation", ("RR",)),
    ("test", ("ER",)),
    ("train", ("E", "EE", "R", "RE")),
    ("validation", ("E", "EE", "R", "RE")),
)


@dataclass(frozen=True)
class _ManifestAuthority:
    relative_file: str
    file_sha256: str
    family_counts: tuple
    sample_counts: tuple
    assignment_sha256: tuple
    partition_templates: tuple


@dataclass(frozen=True)
class _VerifiedManifest:
    family_ids_by_partition: tuple
    family_templates: tuple

    def family_ids(self, partition):
        return dict(self.family_ids_by_partition)[partition]

    def template_by_family(self):
        return dict(self.family_templates)


FROZEN_MANIFEST_AUTHORITY = _ManifestAuthority(
    AUTHORITATIVE_RELATIVE_FILE,
    AUTHORITATIVE_FILE_SHA256,
    FAMILY_COUNTS,
    SAMPLE_COUNTS,
    ASSIGNMENT_SHA256,
    PARTITION_TEMPLATES,
)


def load_train(corpus_dir, family_ids=None):
    """Load all train families or one frozen-gate-compatible sorted subset."""

    return _load_partition(corpus_dir, TRAIN_PARTITION, family_ids)


def load_development(corpus_dir):
    """Load the complete frozen 45-family development assignment."""

    return _load_partition(corpus_dir, DEVELOPMENT_PARTITION, None)


def _load_partition(corpus_dir, partition, family_ids):
    if partition in PROTECTED_PARTITIONS:
        raise GraphEncoderError(
            code="protected_partition_access",
            detail=(
                "{!r} is protected; it requires a separate audited entry point, "
                "never a relaxation of this loader".format(partition)
            ),
        )
    if partition not in LOADABLE_PARTITIONS:
        raise GraphEncoderError(
            code="protected_partition_access",
            detail="GE1 C1 cannot load partition {!r}".format(partition),
        )
    verified = _verify_authoritative_manifest(corpus_dir)
    selected = None
    if partition == TRAIN_PARTITION and family_ids is not None:
        selected = _validate_train_subset(family_ids, verified)
    elif partition == DEVELOPMENT_PARTITION and family_ids is not None:
        raise GraphEncoderError(
            code="protected_partition_access",
            detail="GE1 development subsets are not authorized",
        )
    return load_partition_physical_examples(
        corpus_dir,
        SPLIT_NAME,
        partition,
        selected,
    )


def _validate_train_subset(family_ids, verified):
    try:
        selected = tuple(family_ids)
    except TypeError as exc:
        raise GraphEncoderError(
            "invalid_train_subset", "train family IDs must be iterable"
        ) from exc
    if (
        any(not isinstance(item, str) or not item for item in selected)
        or len(selected) != len(set(selected))
        or selected != tuple(sorted(selected))
    ):
        raise GraphEncoderError(
            "invalid_train_subset",
            "train family IDs must be nonempty, unique, and sorted",
        )
    if len(selected) not in (4, 32):
        raise GraphEncoderError(
            "invalid_train_subset",
            "train subset must contain exactly 4 or 32 families",
        )
    authoritative = set(verified.family_ids(TRAIN_PARTITION))
    if any(item not in authoritative for item in selected):
        raise GraphEncoderError(
            "invalid_train_subset",
            "every requested family must belong to operation_template.train",
        )
    template_by_family = verified.template_by_family()
    observed = Counter(template_by_family[item] for item in selected)
    expected_per_template = 1 if len(selected) == 4 else 8
    expected = Counter({
        "E": expected_per_template,
        "R": expected_per_template,
        "EE": expected_per_template,
        "RE": expected_per_template,
    })
    if observed != expected:
        raise GraphEncoderError(
            "invalid_train_subset",
            "train subset must contain {} families from each of E, R, EE, RE".format(
                expected_per_template
            ),
        )
    return selected


def _verify_authoritative_manifest(corpus_dir):
    root = Path(corpus_dir)
    path = root / FROZEN_MANIFEST_AUTHORITY.relative_file
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise GraphEncoderError(
            "manifest_authority_failure",
            "authoritative manifest is absent: {}".format(
                FROZEN_MANIFEST_AUTHORITY.relative_file
            ),
        ) from exc
    except OSError as exc:
        raise GraphEncoderError(
            "manifest_authority_failure",
            "authoritative manifest is unreadable: {}".format(
                FROZEN_MANIFEST_AUTHORITY.relative_file
            ),
        ) from exc
    return _verify_manifest_bytes(raw, FROZEN_MANIFEST_AUTHORITY)


def _verify_manifest_bytes(raw, authority):
    if not isinstance(raw, bytes):
        raise GraphEncoderError(
            "manifest_authority_failure", "manifest content must be bytes"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != authority.file_sha256:
        raise GraphEncoderError(
            "manifest_authority_failure", "authoritative manifest SHA-256 mismatch"
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphEncoderError(
            "manifest_authority_failure", "authoritative manifest JSON is invalid"
        ) from exc
    if not isinstance(manifest, dict):
        _manifest_failure("manifest root must be an object")
    if manifest.get("name") != SPLIT_NAME:
        _manifest_failure("manifest name must equal operation_template")
    if manifest.get("authoritative_assignment_unit") != "source_family_id":
        _manifest_failure("manifest assignment unit must equal source_family_id")

    expected_family_counts = dict(authority.family_counts)
    if manifest.get("partition_counts") != expected_family_counts:
        _manifest_failure("declared family partition counts differ from C0")
    families = manifest.get("families")
    samples = manifest.get("samples")
    if not isinstance(families, list):
        _manifest_failure("manifest families must be a list")
    if not isinstance(samples, list):
        _manifest_failure("manifest samples must be a list")

    family_ids_by_partition = {
        partition: [] for partition in expected_family_counts
    }
    family_templates = {}
    family_sample_ids = {}
    allowed_templates = {
        partition: set(templates)
        for partition, templates in authority.partition_templates
    }
    for item in families:
        if not isinstance(item, dict):
            _manifest_failure("family assignment records must be objects")
        family_id = item.get("source_family_id")
        partition = item.get("partition")
        template = item.get("operation_template")
        declared_sample_ids = item.get("sample_ids")
        if not isinstance(family_id, str) or not family_id:
            _manifest_failure("family IDs must be nonempty strings")
        if family_id in family_templates:
            _manifest_failure("family IDs must be unique")
        if partition not in family_ids_by_partition:
            _manifest_failure("family partition is not frozen")
        if template not in allowed_templates[partition]:
            _manifest_failure("operation template is assigned to the wrong partition")
        if (
            not isinstance(declared_sample_ids, list)
            or len(declared_sample_ids) != 2
            or any(not isinstance(value, str) or not value for value in declared_sample_ids)
            or len(set(declared_sample_ids)) != 2
        ):
            _manifest_failure("each family must declare two unique sample IDs")
        family_ids_by_partition[partition].append(family_id)
        family_templates[family_id] = template
        family_sample_ids[family_id] = tuple(sorted(declared_sample_ids))

    observed_family_counts = {
        partition: len(values)
        for partition, values in family_ids_by_partition.items()
    }
    if observed_family_counts != expected_family_counts:
        _manifest_failure("recomputed family partition counts differ from C0")
    if len(family_templates) != sum(expected_family_counts.values()):
        _manifest_failure("total family count differs from C0")

    expected_sample_counts = dict(authority.sample_counts)
    sample_counts = Counter()
    sample_ids = set()
    observed_samples_by_family = {}
    for item in samples:
        if not isinstance(item, dict):
            _manifest_failure("sample assignment records must be objects")
        sample_id = item.get("sample_id")
        family_id = item.get("source_family_id")
        partition = item.get("partition")
        template = item.get("operation_template")
        if not isinstance(sample_id, str) or not sample_id:
            _manifest_failure("sample IDs must be nonempty strings")
        if sample_id in sample_ids:
            _manifest_failure("sample IDs must be unique")
        sample_ids.add(sample_id)
        if family_id not in family_templates:
            _manifest_failure("sample references an unknown family")
        family_partition = next(
            assigned
            for assigned, values in family_ids_by_partition.items()
            if family_id in values
        )
        if partition != family_partition:
            _manifest_failure("sample partition disagrees with its family")
        if template != family_templates[family_id]:
            _manifest_failure("sample template disagrees with its family")
        sample_counts[partition] += 1
        observed_samples_by_family.setdefault(family_id, []).append(sample_id)

    if dict(sample_counts) != expected_sample_counts:
        _manifest_failure("recomputed sample partition counts differ from C0")
    if len(sample_ids) != sum(expected_sample_counts.values()):
        _manifest_failure("total sample count differs from C0")
    for family_id, declared in family_sample_ids.items():
        observed = tuple(sorted(observed_samples_by_family.get(family_id, ())))
        if observed != declared:
            _manifest_failure("family and sample assignment metadata disagree")

    expected_hashes = dict(authority.assignment_sha256)
    for partition, family_ids in family_ids_by_partition.items():
        ordered = tuple(sorted(family_ids))
        if _assignment_sha256(ordered) != expected_hashes[partition]:
            _manifest_failure(
                "{} assignment SHA-256 mismatch".format(partition)
            )

    return _VerifiedManifest(
        tuple(
            (partition, tuple(sorted(family_ids_by_partition[partition])))
            for partition in sorted(family_ids_by_partition)
        ),
        tuple(sorted(family_templates.items())),
    )


def _assignment_sha256(family_ids):
    payload = ("\n".join(sorted(family_ids)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_failure(detail):
    raise GraphEncoderError("manifest_authority_failure", detail)
