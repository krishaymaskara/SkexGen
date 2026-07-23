"""Lazy candidate enumeration and deterministic endpoint-disjoint matching."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import math
from typing import Any, Iterable

from prototype.controlled_data.factors import PhysicalSource, factor_blocks
from prototype.controlled_data.feasibility import (
    direction_relation,
    evaluate_feasibility,
    extent_order_relation,
)
from prototype.representation.model import BooleanMode, Direction

from .config import (
    ORIENTATION_POLICY_VERSION,
    CounterfactualConfig,
    EditConfigurationError,
    canonical_json_bytes,
    primary_token_declarations,
    required_minimum,
)
from .edits import before_after, extent_band, target_for
from .identity import (
    endpoint_source_family_id,
    physical_source_bytes,
)
from .model import (
    EditCandidate,
    EditTarget,
    EditType,
    IdentifiedCandidate,
    OrientedCandidate,
)


class CoverageSelectionError(EditConfigurationError):
    def __init__(
        self,
        derived_minimum: int,
        missing_primary_tokens: Iterable[tuple[Any, ...]] = (),
        missing_secondary_tokens: Iterable[tuple[Any, ...]] = (),
        blocked_primary_groups: Iterable[tuple[Any, ...]] = (),
        available_candidates: dict[tuple[Any, ...], int] | None = None,
    ):
        self.derived_minimum = derived_minimum
        self.missing_primary_tokens = tuple(sorted(missing_primary_tokens, key=repr))
        self.missing_secondary_tokens = tuple(sorted(missing_secondary_tokens, key=repr))
        self.blocked_primary_groups = tuple(sorted(blocked_primary_groups, key=repr))
        self.available_candidates = dict(
            sorted((available_candidates or {}).items(), key=lambda item: repr(item[0]))
        )
        super().__init__(
            "counterfactual coverage failed: "
            f"derived_minimum={derived_minimum}, "
            f"missing_primary={list(self.missing_primary_tokens)!r}, "
            f"missing_secondary={list(self.missing_secondary_tokens)!r}, "
            f"blocked_primary={list(self.blocked_primary_groups)!r}"
        )


def iter_raw_candidates(config: CounterfactualConfig):
    """Yield every undirected edit edge exactly once, before feasibility filtering."""

    factor_config = config.factor_config
    for block in factor_blocks(factor_config):
        for index in range(block.size):
            source = block.decode(index)
            for position, operation in enumerate(block.template.operations):
                extent_index = factor_config.sketch_extents.index(
                    source.sketch_extents[position]
                )
                if extent_index + 1 < len(factor_config.sketch_extents):
                    values = list(source.sketch_extents)
                    values[position] = factor_config.sketch_extents[extent_index + 1]
                    changed_extents = tuple(values)
                    changed_band = extent_band(changed_extents, factor_config)
                    if changed_band is source.extent_band:
                        yield EditCandidate(
                            EditType.PROFILE_EXTENT,
                            target_for(EditType.PROFILE_EXTENT, source, position),
                            source,
                            replace(
                                source,
                                sketch_extents=changed_extents,
                                configured_extent_band=changed_band,
                            ),
                        )

                domain = (
                    factor_config.extrusion_distances
                    if operation == "extrude"
                    else factor_config.revolution_angles
                )
                parameter_index = domain.index(source.operation_parameters[position])
                if parameter_index + 1 < len(domain):
                    values = list(source.operation_parameters)
                    values[position] = domain[parameter_index + 1]
                    edit_type = (
                        EditType.EXTRUSION_DISTANCE
                        if operation == "extrude"
                        else EditType.REVOLVE_ANGLE
                    )
                    yield EditCandidate(
                        edit_type,
                        target_for(edit_type, source, position),
                        source,
                        replace(source, operation_parameters=tuple(values)),
                    )

                if source.directions[position] is Direction.POSITIVE:
                    values = list(source.directions)
                    values[position] = Direction.NEGATIVE
                    yield EditCandidate(
                        EditType.OPERATION_DIRECTION,
                        target_for(EditType.OPERATION_DIRECTION, source, position),
                        source,
                        replace(source, directions=tuple(values)),
                    )

            if (
                source.history_depth == 2
                and source.later_boolean_mode is BooleanMode.JOIN
            ):
                yield EditCandidate(
                    EditType.BOOLEAN_MODE,
                    target_for(EditType.BOOLEAN_MODE, source, 1),
                    source,
                    replace(source, later_boolean_mode=BooleanMode.CUT),
                )


def iter_accepted_candidates(config: CounterfactualConfig):
    for candidate in iter_raw_candidates(config):
        if (
            evaluate_feasibility(candidate.endpoint_a).accepted
            and evaluate_feasibility(candidate.endpoint_b).accepted
        ):
            yield candidate


def candidate_counts(config: CounterfactualConfig) -> dict[str, Any]:
    raw = Counter()
    accepted = Counter()
    cells = Counter()
    for candidate in iter_raw_candidates(config):
        raw[candidate.edit_type.value] += 1
        if (
            evaluate_feasibility(candidate.endpoint_a).accepted
            and evaluate_feasibility(candidate.endpoint_b).accepted
        ):
            accepted[candidate.edit_type.value] += 1
            cells[primary_token(candidate)] += 1
    raw_total = sum(raw.values())
    accepted_total = sum(accepted.values())
    return {
        "raw_undirected": raw_total,
        "raw_directed": 2 * raw_total,
        "accepted_undirected": accepted_total,
        "accepted_directed": 2 * accepted_total,
        "infeasible_undirected": raw_total - accepted_total,
        "duplicate_reversed_removed": accepted_total,
        "raw_by_edit_type": dict(sorted(raw.items())),
        "accepted_by_edit_type": dict(sorted(accepted.items())),
        "accepted_by_primary_token": {
            _token_text(key): cells[key] for key in sorted(cells)
        },
    }


def primary_token(candidate: EditCandidate) -> tuple[str, str, int, str]:
    return (
        candidate.edit_type.value,
        candidate.endpoint_a.operation_template.value,
        candidate.target.operation_index,
        candidate.endpoint_a.extent_band.value,
    )


def declared_secondary_tokens(config: CounterfactualConfig) -> frozenset[tuple[Any, ...]]:
    tokens: set[tuple[Any, ...]] = set()
    for edit_type in tuple(EditType):
        prefix = edit_type.value
        tokens.update(
            (prefix, "primitive_family", value)
            for value in ("rectangle_lines", "circle", "capsule_line_arc")
        )
        tokens.update(
            (prefix, "reference_plane", value) for value in ("XY", "XZ", "YZ")
        )
        tokens.update((prefix, "boolean_mode", value) for value in ("join", "cut"))
        tokens.update(
            (prefix, "direction", value) for value in ("positive", "negative")
        )
        tokens.update(
            (prefix, "direction_relation", value) for value in ("same", "opposite")
        )
        tokens.update(
            (prefix, "extent_relation", value)
            for value in ("smaller", "equal", "larger")
        )
    domains = (
        (EditType.PROFILE_EXTENT, config.factor_config.sketch_extents),
        (EditType.EXTRUSION_DISTANCE, config.factor_config.extrusion_distances),
        (EditType.REVOLVE_ANGLE, config.factor_config.revolution_angles),
    )
    for edit_type, domain in domains:
        tokens.update(
            (edit_type.value, "parameter_value", float(value)) for value in domain
        )
        tokens.update(
            (edit_type.value, "edit_direction", value)
            for value in ("increase", "decrease")
        )
    tokens.update(
        (EditType.OPERATION_DIRECTION.value, "edit_direction", value)
        for value in ("positive_to_negative", "negative_to_positive")
    )
    tokens.update(
        (EditType.BOOLEAN_MODE.value, "edit_direction", value)
        for value in ("join_to_cut", "cut_to_join")
    )
    return frozenset(tokens)


def candidate_secondary_capabilities(
    candidate: EditCandidate,
) -> frozenset[tuple[Any, ...]]:
    edit_type = candidate.edit_type.value
    tokens: set[tuple[Any, ...]] = {
        (edit_type, "primitive_family", candidate.endpoint_a.primitive_family.value),
        (edit_type, "reference_plane", candidate.endpoint_a.reference_plane.value),
    }
    for endpoint in (candidate.endpoint_a, candidate.endpoint_b):
        if endpoint.later_boolean_mode is not None:
            tokens.add((edit_type, "boolean_mode", endpoint.later_boolean_mode.value))
        tokens.update(
            (edit_type, "direction", direction.value)
            for direction in endpoint.directions
        )
        if endpoint.history_depth == 2:
            tokens.add(
                (edit_type, "direction_relation", direction_relation(endpoint))
            )
            tokens.add((edit_type, "extent_relation", extent_order_relation(endpoint)))
    position = candidate.target.operation_index
    if candidate.edit_type is EditType.PROFILE_EXTENT:
        values = (
            candidate.endpoint_a.sketch_extents[position],
            candidate.endpoint_b.sketch_extents[position],
        )
        tokens.update((edit_type, "parameter_value", value) for value in values)
        tokens.update(
            (edit_type, "edit_direction", value)
            for value in ("increase", "decrease")
        )
    elif candidate.edit_type in (
        EditType.EXTRUSION_DISTANCE,
        EditType.REVOLVE_ANGLE,
    ):
        values = (
            candidate.endpoint_a.operation_parameters[position],
            candidate.endpoint_b.operation_parameters[position],
        )
        tokens.update((edit_type, "parameter_value", value) for value in values)
        tokens.update(
            (edit_type, "edit_direction", value)
            for value in ("increase", "decrease")
        )
    elif candidate.edit_type is EditType.OPERATION_DIRECTION:
        tokens.update(
            (edit_type, "edit_direction", value)
            for value in ("positive_to_negative", "negative_to_positive")
        )
    else:
        tokens.update(
            (edit_type, "edit_direction", value)
            for value in ("join_to_cut", "cut_to_join")
        )
    return frozenset(tokens)


def identify_candidate(candidate: EditCandidate) -> IdentifiedCandidate:
    left_id = endpoint_source_family_id(candidate.endpoint_a)
    right_id = endpoint_source_family_id(candidate.endpoint_b)
    if left_id == right_id:
        raise EditConfigurationError("distinct candidate endpoints share a source-family ID")
    if left_id < right_id:
        ordered = candidate
        lower, higher = left_id, right_id
    else:
        ordered = replace(
            candidate,
            endpoint_a=candidate.endpoint_b,
            endpoint_b=candidate.endpoint_a,
        )
        lower, higher = right_id, left_id
    return IdentifiedCandidate(ordered, lower, higher)


def undirected_key(candidate: IdentifiedCandidate) -> tuple[Any, ...]:
    item = candidate.candidate
    return (
        item.edit_type.value,
        item.target.operation_index,
        item.target.node_id,
        item.target.changed_field,
        candidate.lower_source_family_id,
        candidate.higher_source_family_id,
    )


def canonical_undirected_bytes(candidate: IdentifiedCandidate) -> bytes:
    item = candidate.candidate
    return canonical_json_bytes(
        {
            "edit_type": item.edit_type.value,
            "target": item.target.to_dict(),
            "lower_source_family_id": candidate.lower_source_family_id,
            "higher_source_family_id": candidate.higher_source_family_id,
        }
    )


def select_oriented_candidates(
    config: CounterfactualConfig,
    *,
    accepted_candidates: Iterable[EditCandidate] | None = None,
) -> tuple[OrientedCandidate, ...]:
    config.validate()
    if config.num_edit_families < required_minimum():
        raise CoverageSelectionError(
            required_minimum(),
            missing_primary_tokens=primary_token_declarations(),
            missing_secondary_tokens=declared_secondary_tokens(config),
        )
    groups: dict[tuple[str, str, int, str], list[IdentifiedCandidate]] = defaultdict(list)
    all_candidates: list[IdentifiedCandidate] = []
    seen_keys: set[tuple[Any, ...]] = set()
    interned_sources: dict[PhysicalSource, PhysicalSource] = {}
    iterable = accepted_candidates or iter_accepted_candidates(config)
    for raw in iterable:
        endpoint_a = interned_sources.setdefault(raw.endpoint_a, raw.endpoint_a)
        endpoint_b = interned_sources.setdefault(raw.endpoint_b, raw.endpoint_b)
        if endpoint_a is not raw.endpoint_a or endpoint_b is not raw.endpoint_b:
            raw = replace(raw, endpoint_a=endpoint_a, endpoint_b=endpoint_b)
        identified = identify_candidate(raw)
        key = undirected_key(identified)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        groups[primary_token(identified.candidate)].append(identified)
        all_candidates.append(identified)

    declared_primary = set(primary_token_declarations())
    actual_primary = set(groups)
    missing_primary = declared_primary - actual_primary
    undeclared_primary = actual_primary - declared_primary
    if missing_primary or undeclared_primary:
        raise CoverageSelectionError(
            required_minimum(),
            missing_primary_tokens=missing_primary,
            blocked_primary_groups=undeclared_primary,
            available_candidates={key: len(value) for key, value in groups.items()},
        )

    required_secondary = declared_secondary_tokens(config)
    covered: set[tuple[Any, ...]] = set()
    used_endpoints: set[str] = set()
    selected: list[IdentifiedCandidate] = []
    blocked: list[tuple[Any, ...]] = []
    ordered_groups = sorted(
        declared_primary,
        key=lambda key: (len(groups[key]), canonical_json_bytes(list(key))),
    )
    for group_key in ordered_groups:
        best = None
        best_gain = -1
        best_bytes = None
        for candidate in groups[group_key]:
            if (
                candidate.lower_source_family_id in used_endpoints
                or candidate.higher_source_family_id in used_endpoints
            ):
                continue
            gain = len(
                (candidate_secondary_capabilities(candidate.candidate) & required_secondary)
                - covered
            )
            canonical = canonical_undirected_bytes(candidate)
            if (
                gain > best_gain
                or (gain == best_gain and (best_bytes is None or canonical < best_bytes))
            ):
                best = candidate
                best_gain = gain
                best_bytes = canonical
        if best is None:
            blocked.append(group_key)
            continue
        selected.append(best)
        used_endpoints.update(
            (best.lower_source_family_id, best.higher_source_family_id)
        )
        covered.update(
            candidate_secondary_capabilities(best.candidate) & required_secondary
        )

    missing_secondary = required_secondary - covered
    if blocked or missing_secondary or len(selected) != required_minimum():
        raise CoverageSelectionError(
            required_minimum(),
            missing_secondary_tokens=missing_secondary,
            blocked_primary_groups=blocked,
            available_candidates={key: len(groups[key]) for key in blocked},
        )

    anchor_keys = {undirected_key(item) for item in selected}
    if config.num_edit_families > len(selected):
        positions = _affine_positions(len(all_candidates), config)
        for position in positions:
            if len(selected) >= config.num_edit_families:
                break
            candidate = all_candidates[position]
            if undirected_key(candidate) in anchor_keys:
                continue
            if (
                candidate.lower_source_family_id in used_endpoints
                or candidate.higher_source_family_id in used_endpoints
            ):
                continue
            selected.append(candidate)
            anchor_keys.add(undirected_key(candidate))
            used_endpoints.update(
                (candidate.lower_source_family_id, candidate.higher_source_family_id)
            )
        if len(selected) != config.num_edit_families:
            raise CoverageSelectionError(
                required_minimum(),
                blocked_primary_groups=(("filler_selection_exhausted",),),
            )

    anchors = _orient_anchors(tuple(selected[:required_minimum()]))
    fillers = tuple(
        _hash_orient(candidate)
        for candidate in selected[required_minimum():]
    )
    result = anchors + fillers
    _verify_selection(result, config)
    return result


def _orient_anchors(
    candidates: tuple[IdentifiedCandidate, ...],
) -> tuple[OrientedCandidate, ...]:
    by_type: dict[EditType, list[IdentifiedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.candidate.edit_type].append(candidate)
    result = []
    forward_labels = {
        EditType.PROFILE_EXTENT: "increase",
        EditType.EXTRUSION_DISTANCE: "increase",
        EditType.REVOLVE_ANGLE: "increase",
        EditType.OPERATION_DIRECTION: "positive_to_negative",
        EditType.BOOLEAN_MODE: "join_to_cut",
    }
    for edit_type in tuple(EditType):
        items = sorted(
            by_type[edit_type],
            key=lambda item: (
                hashlib.sha256(canonical_undirected_bytes(item)).digest(),
                canonical_undirected_bytes(item),
            ),
        )
        forward_count = len(items) // 2 + len(items) % 2
        for index, candidate in enumerate(items):
            label = (
                forward_labels[edit_type]
                if index < forward_count
                else _opposite_label(forward_labels[edit_type])
            )
            base, edited = _orient_for_label(candidate.candidate, label)
            result.append(OrientedCandidate(candidate, base, edited, True))
    return tuple(sorted(result, key=lambda item: canonical_undirected_bytes(item.identified)))


def _hash_orient(candidate: IdentifiedCandidate) -> OrientedCandidate:
    item = candidate.candidate
    descriptor = {
        "orientation_policy_version": ORIENTATION_POLICY_VERSION,
        "candidate": {
            "edit_type": item.edit_type.value,
            "target": item.target.to_dict(),
            "lower_source_family_id": candidate.lower_source_family_id,
            "higher_source_family_id": candidate.higher_source_family_id,
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor)).digest()
    if int.from_bytes(digest, "big") % 2 == 0:
        base, edited = item.endpoint_a, item.endpoint_b
    else:
        base, edited = item.endpoint_b, item.endpoint_a
    return OrientedCandidate(candidate, base, edited, False)


def orientation_label(item: OrientedCandidate) -> str:
    candidate = item.identified.candidate
    position = candidate.target.operation_index
    before, after = before_after(
        candidate.edit_type, item.base_source, item.edited_source, position
    )
    if candidate.edit_type in (
        EditType.PROFILE_EXTENT,
        EditType.EXTRUSION_DISTANCE,
        EditType.REVOLVE_ANGLE,
    ):
        return "increase" if after > before else "decrease"
    return f"{before}_to_{after}"


def _orient_for_label(
    candidate: EditCandidate, label: str
) -> tuple[PhysicalSource, PhysicalSource]:
    current = OrientedCandidate(
        IdentifiedCandidate(candidate, "", ""),
        candidate.endpoint_a,
        candidate.endpoint_b,
        True,
    )
    if orientation_label(current) == label:
        return candidate.endpoint_a, candidate.endpoint_b
    reverse = OrientedCandidate(
        current.identified,
        candidate.endpoint_b,
        candidate.endpoint_a,
        True,
    )
    if orientation_label(reverse) != label:
        raise EditConfigurationError(f"candidate cannot realize orientation {label!r}")
    return candidate.endpoint_b, candidate.endpoint_a


def _opposite_label(label: str) -> str:
    pairs = {
        "increase": "decrease",
        "positive_to_negative": "negative_to_positive",
        "join_to_cut": "cut_to_join",
    }
    return pairs[label]


def _affine_positions(size: int, config: CounterfactualConfig) -> tuple[int, ...]:
    if size <= 0:
        return ()
    digest = hashlib.sha256(
        b"counterfactual-filler-affine-v1\0"
        + config.canonical_bytes()
        + b"\0"
        + str(config.seed).encode("ascii")
    ).digest()
    multiplier = int.from_bytes(digest[:16], "big") % size or 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size or 1
    offset = int.from_bytes(digest[16:], "big") % size
    return tuple((multiplier * position + offset) % size for position in range(size))


def _verify_selection(
    selected: tuple[OrientedCandidate, ...], config: CounterfactualConfig
) -> None:
    endpoints = [
        endpoint
        for item in selected
        for endpoint in (
            item.identified.lower_source_family_id,
            item.identified.higher_source_family_id,
        )
    ]
    if len(endpoints) != len(set(endpoints)):
        raise EditConfigurationError("selected edit families reuse an endpoint")
    anchors = tuple(item for item in selected if item.is_coverage_anchor)
    if len(anchors) != required_minimum():
        raise EditConfigurationError("coverage-anchor count is not the derived minimum")
    if {primary_token(item.identified.candidate) for item in anchors} != set(
        primary_token_declarations()
    ):
        raise EditConfigurationError("coverage anchors do not cover every primary token")
    covered: set[tuple[Any, ...]] = set()
    for item in anchors:
        covered.update(candidate_secondary_capabilities(item.identified.candidate))
    missing = declared_secondary_tokens(config) - covered
    if missing:
        raise EditConfigurationError(f"coverage anchors miss secondary tokens {missing!r}")
    counts = Counter(
        (item.identified.candidate.edit_type.value, orientation_label(item))
        for item in anchors
    )
    expected = {
        ("profile_extent", "increase"): 10,
        ("profile_extent", "decrease"): 10,
        ("extrusion_distance", "increase"): 5,
        ("extrusion_distance", "decrease"): 5,
        ("revolve_angle", "increase"): 5,
        ("revolve_angle", "decrease"): 5,
        ("operation_direction", "positive_to_negative"): 10,
        ("operation_direction", "negative_to_positive"): 10,
        ("boolean_mode", "join_to_cut"): 4,
        ("boolean_mode", "cut_to_join"): 4,
    }
    if counts != expected:
        raise EditConfigurationError(f"anchor orientation quotas differ: {dict(counts)!r}")


def _token_text(token: tuple[Any, ...]) -> str:
    return "|".join(str(item) for item in token)
