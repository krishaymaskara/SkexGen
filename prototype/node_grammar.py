"""Immutable prefix grammar for controlled flat CAD node sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from prototype.model_data.vocab import NODE_TYPES


NODE_GRAMMAR_CONTRACT_ID = "controlled-prefix-node-grammar-v5"
COMPLETION_ALGORITHM_ID = "exhaustive-finite-prefix-completion-v1"
VALID_REQUESTED_NODE_COUNTS = (4, 5, 7, 8, 9)
CONTROLLED_OPERATION_LIMIT = 2


class NodeGrammarError(ValueError):
    """A V5 grammar input or continuation is invalid."""

    def __init__(self, code, detail, context=None):
        self.code = code
        self.detail = detail
        self.context = dict(context or {})
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class NodeGrammarContract:
    contract_id: str
    node_vocabulary: tuple
    start_node: str
    transitions: tuple
    terminal_nodes: tuple
    valid_requested_node_counts: tuple
    operation_limit: int
    completion_algorithm_id: str

    def to_dict(self):
        values = asdict(self)
        values["node_vocabulary"] = list(self.node_vocabulary)
        values["transitions"] = [list(item) for item in self.transitions]
        values["terminal_nodes"] = list(self.terminal_nodes)
        values["valid_requested_node_counts"] = list(
            self.valid_requested_node_counts
        )
        return values

    def to_json(self):
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


V5_NODE_GRAMMAR = NodeGrammarContract(
    NODE_GRAMMAR_CONTRACT_ID,
    tuple(NODE_TYPES.tokens),
    "reference_plane",
    (
        ("START", "reference_plane"),
        ("reference_plane", "sketch"),
        ("sketch", "profile"),
        ("profile", "extrude"),
        ("profile", "axis"),
        ("axis", "revolve"),
        ("extrude", "sketch"),
        ("revolve", "sketch"),
    ),
    ("extrude", "revolve"),
    VALID_REQUESTED_NODE_COUNTS,
    CONTROLLED_OPERATION_LIMIT,
    COMPLETION_ALGORITHM_ID,
)


def validate_node_grammar_contract(contract=V5_NODE_GRAMMAR):
    if not isinstance(contract, NodeGrammarContract):
        raise NodeGrammarError(
            "invalid_v5_grammar_contract", "contract has the wrong type"
        )
    expected = V5_NODE_GRAMMAR
    if contract != expected or contract.node_vocabulary != tuple(NODE_TYPES.tokens):
        raise NodeGrammarError(
            "invalid_v5_grammar_contract",
            "grammar metadata differs from the frozen V5 contract",
        )
    return contract


def enumerate_complete_node_sequences(contract=V5_NODE_GRAMMAR):
    """Enumerate the finite language from transitions and operation limits."""

    validate_node_grammar_contract(contract)
    transitions = {}
    for source, target in contract.transitions:
        transitions.setdefault(source, []).append(target)
    completed = []

    def visit(prefix, state, operation_count):
        if (
            state in contract.terminal_nodes
            and 1 <= operation_count <= contract.operation_limit
        ):
            completed.append(prefix)
        if operation_count >= contract.operation_limit:
            return
        for candidate in transitions.get(state, ()):
            next_count = operation_count + int(
                candidate in contract.terminal_nodes
            )
            visit(prefix + (candidate,), candidate, next_count)

    visit((), "START", 0)
    result = tuple(sorted(set(completed), key=lambda item: (len(item), item)))
    if tuple(sorted(set(map(len, result)))) != contract.valid_requested_node_counts:
        raise NodeGrammarError(
            "invalid_v5_grammar_contract",
            "enumerated lengths disagree with contract metadata",
        )
    return result


def legal_next_node_ids(prefix, requested_node_count, contract=V5_NODE_GRAMMAR):
    """Return candidates that retain an exact-length terminal completion."""

    validate_node_grammar_contract(contract)
    prefix = _validated_prefix(prefix)
    requested_node_count = _validated_requested_count(
        requested_node_count, contract
    )
    if len(prefix) >= requested_node_count:
        prefix_tokens = tuple(NODE_TYPES.tokens[item] for item in prefix)
        raise NodeGrammarError(
            "no_valid_node_grammar_continuation",
            "the requested sequence is already complete",
            _context(
                prefix,
                requested_node_count,
                _transition_candidates(prefix_tokens, contract),
            ),
        )
    all_sequences = enumerate_complete_node_sequences(contract)
    prefix_tokens = tuple(NODE_TYPES.tokens[item] for item in prefix)
    structurally_matching = tuple(
        sequence
        for sequence in all_sequences
        if sequence[:len(prefix)] == prefix_tokens
    )
    matching = tuple(
        sequence for sequence in structurally_matching
        if len(sequence) == requested_node_count
    )
    if not matching:
        candidates = _transition_candidates(prefix_tokens, contract)
        code = (
            "no_valid_node_grammar_continuation"
            if structurally_matching else "invalid_v5_generated_prefix"
        )
        raise NodeGrammarError(
            code,
            "prefix is not extendable under the requested length",
            _context(prefix, requested_node_count, candidates),
        )
    candidates = tuple(sorted({NODE_TYPES.id(item[len(prefix)]) for item in matching}))
    if not candidates:
        raise NodeGrammarError(
            "no_valid_node_grammar_continuation",
            "prefix has no exact-length continuation",
            _context(prefix, requested_node_count, ()),
        )
    return candidates


def validate_complete_node_sequence(
    node_ids, requested_node_count, contract=V5_NODE_GRAMMAR
):
    validate_node_grammar_contract(contract)
    node_ids = _validated_prefix(node_ids)
    requested_node_count = _validated_requested_count(
        requested_node_count, contract
    )
    tokens = tuple(NODE_TYPES.tokens[item] for item in node_ids)
    valid = set(enumerate_complete_node_sequences(contract))
    if len(tokens) != requested_node_count or tokens not in valid:
        raise NodeGrammarError(
            "invalid_v5_generated_prefix",
            "node sequence is not a complete controlled program",
            _context(node_ids, requested_node_count, ()),
        )
    return tokens


def grammar_state_evidence(prefix, requested_node_count, legal_candidates):
    prefix = _validated_prefix(prefix)
    tokens = tuple(NODE_TYPES.tokens[item] for item in prefix)
    last = "START" if not tokens else tokens[-1]
    operation_count = sum(item in ("extrude", "revolve") for item in tokens)
    return {
        "state": "{}:operations={}".format(last, operation_count),
        "requested_node_count": requested_node_count,
        "current_position": len(prefix),
        "remaining_positions": requested_node_count - len(prefix),
        "constrained_prefix": list(prefix),
        "legal_node_type_ids": list(legal_candidates),
    }


def node_grammar_metadata(contract=V5_NODE_GRAMMAR):
    validate_node_grammar_contract(contract)
    return contract.to_dict()


def _validated_requested_count(value, contract):
    if isinstance(value, bool) or not isinstance(value, int):
        raise NodeGrammarError(
            "invalid_v5_requested_node_count",
            "requested node count must be an integer",
        )
    if value not in contract.valid_requested_node_counts:
        raise NodeGrammarError(
            "invalid_v5_requested_node_count",
            "requested node count is not supported by the controlled grammar",
        )
    return value


def _validated_prefix(prefix):
    if not isinstance(prefix, (tuple, list)):
        raise NodeGrammarError(
            "invalid_v5_generated_prefix", "prefix must be a tuple or list"
        )
    result = tuple(prefix)
    for value in result:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(NODE_TYPES.tokens)
            or value in (NODE_TYPES.pad_id, NODE_TYPES.id(None))
        ):
            raise NodeGrammarError(
                "invalid_v5_generated_prefix",
                "active prefix contains an invalid or sentinel node ID",
            )
    return result


def _context(prefix, requested_node_count, candidates):
    evidence = grammar_state_evidence(prefix, requested_node_count, candidates)
    evidence["legal_transition_candidates_before_completion_filtering"] = list(
        candidates
    )
    return evidence


def _transition_candidates(prefix_tokens, contract):
    state = "START" if not prefix_tokens else prefix_tokens[-1]
    return tuple(
        NODE_TYPES.id(target)
        for source, target in contract.transitions
        if source == state
    )
