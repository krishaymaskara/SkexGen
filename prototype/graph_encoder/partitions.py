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

C7_SELECTION_VERSION = "GE1-C7-SUFFICIENCY-v1"
C7_ACCESSIBLE_TEMPLATES = ("E", "R", "EE", "RE")
C7_TINY_PER_TEMPLATE = 1
C7_SCALED_PER_TEMPLATE = 8


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


@dataclass(frozen=True)
class C7SufficiencySelection:
    """Payload-free deterministic C7 tiny and scaled train cohorts."""

    version: str
    templates: tuple
    tiny_family_ids: tuple
    scaled_family_ids: tuple
    tiny_family_ids_sha256: str
    scaled_family_ids_sha256: str
    selected_templates: tuple
    ranking_records: tuple

    def to_dict(self):
        return {
            "version": self.version,
            "manifest_name": SPLIT_NAME,
            "manifest_sha256": AUTHORITATIVE_FILE_SHA256,
            "partition": TRAIN_PARTITION,
            "templates": list(self.templates),
            "ranking_algorithm": "sha256_utf8_hex_then_source_family_id",
            "ranking_hash_material": (
                "GE1-C7-SUFFICIENCY-v1\\n"
                "<operation_template>\\n<source_family_id>\\n"
            ),
            "ranking_hash_material_encoding": "UTF-8",
            "ranking_hash_material_line_ending": "LF_with_final_LF",
            "tiny_per_template": C7_TINY_PER_TEMPLATE,
            "scaled_per_template": C7_SCALED_PER_TEMPLATE,
            "tiny_family_ids": list(self.tiny_family_ids),
            "scaled_family_ids": list(self.scaled_family_ids),
            "tiny_family_ids_sha256": self.tiny_family_ids_sha256,
            "scaled_family_ids_sha256": self.scaled_family_ids_sha256,
            "tiny_nested_in_scaled": set(self.tiny_family_ids).issubset(
                self.scaled_family_ids
            ),
            "selected_templates": dict(self.selected_templates),
            "ranking_records": [
                {
                    "operation_template": template,
                    "source_family_id": family_id,
                    "rank_sha256": rank,
                }
                for template, family_id, rank in self.ranking_records
            ],
            "metadata_only": True,
            "cad_history_payload_accessed": False,
            "development_accessed": False,
            "systematic_rr_accessed": False,
            "test_er_accessed": False,
            "iid_accessed": False,
            "history_depth_accessed": False,
            "geometry_extrapolation_accessed": False,
        }


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


def c7_family_rank(operation_template, source_family_id):
    """Return the exact accepted metadata-only C7 family rank."""

    if operation_template not in C7_ACCESSIBLE_TEMPLATES:
        raise GraphEncoderError(
            "invalid_c7_template",
            "C7 template must be one of E, R, EE, RE",
        )
    if not isinstance(source_family_id, str) or not source_family_id:
        raise GraphEncoderError(
            "invalid_c7_family_id", "C7 family ID must be a nonempty string"
        )
    material = (
        C7_SELECTION_VERSION
        + "\n"
        + operation_template
        + "\n"
        + source_family_id
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def selected_family_ids_sha256(family_ids):
    """Hash unique sorted IDs, one UTF-8 ID per LF-terminated line."""

    values = tuple(family_ids)
    if (
        not values
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
        or any(not isinstance(item, str) or not item for item in values)
    ):
        raise GraphEncoderError(
            "invalid_c7_selection",
            "selected family IDs must be nonempty, unique, and sorted",
        )
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def select_c7_sufficiency_subsets(corpus_dir):
    """Verify manifest authority and select C7 cohorts without payload access."""

    verified = _verify_authoritative_manifest(corpus_dir)
    return _select_c7_from_verified(verified)


def _select_c7_from_verified(verified):
    if not isinstance(verified, _VerifiedManifest):
        raise TypeError("verified must be _VerifiedManifest")
    train_ids = verified.family_ids(TRAIN_PARTITION)
    template_by_family = verified.template_by_family()
    candidates = {template: [] for template in C7_ACCESSIBLE_TEMPLATES}
    for family_id in train_ids:
        template = template_by_family.get(family_id)
        if template not in candidates:
            raise GraphEncoderError(
                "invalid_c7_template_composition",
                "operation_template.train contains a non-C7 template",
            )
        candidates[template].append(
            (c7_family_rank(template, family_id), family_id)
        )
    if any(len(candidates[template]) < C7_SCALED_PER_TEMPLATE
           for template in C7_ACCESSIBLE_TEMPLATES):
        raise GraphEncoderError(
            "invalid_c7_template_composition",
            "each C7 train template requires at least eight families",
        )

    tiny = []
    scaled = []
    ranking_records = []
    for template in C7_ACCESSIBLE_TEMPLATES:
        ranked = tuple(sorted(candidates[template]))
        tiny.extend(family_id for unused_rank, family_id in ranked[:1])
        scaled.extend(
            family_id
            for unused_rank, family_id in ranked[:C7_SCALED_PER_TEMPLATE]
        )
        ranking_records.extend(
            (template, family_id, rank)
            for rank, family_id in ranked[:C7_SCALED_PER_TEMPLATE]
        )
    tiny = tuple(sorted(tiny))
    scaled = tuple(sorted(scaled))
    if len(tiny) != 4 or len(set(tiny)) != 4:
        raise GraphEncoderError(
            "invalid_c7_selection", "tiny C7 set must contain four families"
        )
    if len(scaled) != 32 or len(set(scaled)) != 32:
        raise GraphEncoderError(
            "invalid_c7_selection", "scaled C7 set must contain 32 families"
        )
    if not set(tiny).issubset(scaled):
        raise GraphEncoderError(
            "invalid_c7_selection", "tiny C7 set must be nested in scaled"
        )
    selected_templates = tuple(sorted(
        (family_id, template_by_family[family_id]) for family_id in scaled
    ))
    tiny_counts = Counter(template_by_family[item] for item in tiny)
    scaled_counts = Counter(template_by_family[item] for item in scaled)
    if tiny_counts != Counter({item: 1 for item in C7_ACCESSIBLE_TEMPLATES}):
        raise GraphEncoderError(
            "invalid_c7_template_composition",
            "tiny set must contain one family per C7 template",
        )
    if scaled_counts != Counter({item: 8 for item in C7_ACCESSIBLE_TEMPLATES}):
        raise GraphEncoderError(
            "invalid_c7_template_composition",
            "scaled set must contain eight families per C7 template",
        )
    return C7SufficiencySelection(
        C7_SELECTION_VERSION,
        C7_ACCESSIBLE_TEMPLATES,
        tiny,
        scaled,
        selected_family_ids_sha256(tiny),
        selected_family_ids_sha256(scaled),
        selected_templates,
        tuple(ranking_records),
    )


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
