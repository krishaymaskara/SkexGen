"""Deterministic train-only initialization for the mixed EMA codebook."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from .data import make_data_loader


DISTINCT_TOLERANCE = 1e-6
KMEANS_LLOYD_ITERATIONS = 8
PSEUDO_COUNT_POLICY = "cluster-proportions-total-codebook-size"


class VQInitializationError(RuntimeError):
    """The requested codebook initialization cannot be performed safely."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def initialize_train_codebook(
    model,
    training_data,
    training_config,
    device,
    torch_module=torch,
):
    """Initialize every EMA codebook buffer from training examples only."""

    if not training_data.train_examples:
        raise VQInitializationError(
            "empty_initialization_partition",
            "training initialization requires training examples",
        )
    if (
        len(training_data.train_examples)
        != len(training_data.train_family_ids)
        or len(set(training_data.train_family_ids))
        != len(training_data.train_family_ids)
    ):
        raise VQInitializationError(
            "invalid_initialization_authority",
            "training examples require one unique authoritative family ID each",
        )
    if set(training_data.train_family_ids) & set(
        training_data.validation_family_ids
    ):
        raise VQInitializationError(
            "initialization_partition_overlap",
            "training and validation family IDs overlap",
        )
    vectors = collect_train_prequant_vectors(
        model,
        training_data.train_examples,
        training_config.batch_size,
        training_config.dataloader_workers,
        training_config.seed,
        device,
        torch_module,
    )
    (
        centers,
        assignments,
        counts,
        pseudo_counts,
        pseudo_sums,
    ) = initialize_quantizer_from_vectors(
        model.vq,
        vectors,
        training_config.seed,
        torch_module,
    )
    distances = _squared_distances(vectors, centers)
    selected = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
    if not bool(torch_module.isfinite(selected).all().item()):
        raise VQInitializationError(
            "nonfinite_kmeans_inertia",
            "final assigned k-means distances are non-finite",
        )
    first_batch_vector_count = (
        min(training_config.batch_size, len(training_data.train_examples))
        * model.config.latent_tokens
    )
    normal_mass = float(model.vq.num_embeddings)
    treatment_mass = float(pseudo_counts.sum().item())
    first_batch_mass = (
        (1.0 - model.vq.decay) * first_batch_vector_count
    )
    report = {
        "mode": "train-kmeans",
        "method": "seeded-kmeans++-fixed-lloyd",
        "seed": training_config.seed,
        "partition": "train",
        "train_family_count": len(training_data.train_family_ids),
        "validation_family_count_used": 0,
        "test_family_count_used": 0,
        "prequant_vector_count": int(vectors.size(0)),
        "prequant_width": int(vectors.size(1)),
        "distinct_tolerance": DISTINCT_TOLERANCE,
        "distinct_prequant_vector_count": _distinct_count(
            vectors, DISTINCT_TOLERANCE
        ),
        "codebook_size": model.vq.num_embeddings,
        "lloyd_iterations": KMEANS_LLOYD_ITERATIONS,
        "cluster_counts": [
            int(value) for value in counts.to("cpu").tolist()
        ],
        "pseudo_count_policy": PSEUDO_COUNT_POLICY,
        "pseudo_cluster_counts": [
            float(value) for value in pseudo_counts.to("cpu").tolist()
        ],
        "pseudo_count_total": treatment_mass,
        "normal_initial_count_total": normal_mass,
        "effective_mass_ratio_to_normal": treatment_mass / normal_mass,
        "ema_decay": float(model.vq.decay),
        "ema_epsilon": float(model.vq.epsilon),
        "ema_count_update_equation": (
            "n_i'=decay*n_i+(1-decay)*batch_count_i"
        ),
        "ema_weight_update_equation": (
            "w_i'=decay*w_i+(1-decay)*batch_sum_i"
        ),
        "ema_smoothing_equation": (
            "smoothed_i=(n_i'+epsilon)"
            "/(sum_j(n_j')+K*epsilon)*sum_j(n_j')"
        ),
        "embedding_update_equation": (
            "embedding_i=w_i'/max(smoothed_i,epsilon)"
        ),
        "first_minibatch_vector_count": first_batch_vector_count,
        "normal_first_minibatch_new_mass_fraction": (
            first_batch_mass
            / (model.vq.decay * normal_mass + first_batch_mass)
        ),
        "treatment_first_minibatch_new_mass_fraction": (
            first_batch_mass
            / (model.vq.decay * treatment_mass + first_batch_mass)
        ),
        "inertia": float(selected.sum().item()),
        "centers_sha256": _tensor_sha256(model.vq.embedding),
        "ema_cluster_size_sha256": _tensor_sha256(
            model.vq.ema_cluster_size
        ),
        "ema_weight_sha256": _tensor_sha256(model.vq.ema_weight),
        "finite": True,
        "vector_position_semantics": (
            "every unmasked learned latent-query position; exactly "
            "latent_tokens vectors per authoritative training family"
        ),
    }
    report["report_sha256"] = initialization_report_sha256(report)
    return report


def initialize_quantizer_from_vectors(
    quantizer, vectors, seed, torch_module=torch
):
    """Cluster vectors and make embeddings, EMA counts, and EMA sums agree."""

    centers, assignments, counts, sums = deterministic_kmeans(
        vectors,
        quantizer.num_embeddings,
        seed,
        KMEANS_LLOYD_ITERATIONS,
        DISTINCT_TOLERANCE,
        torch_module,
    )
    pseudo_counts = (
        counts
        / counts.sum()
        * float(quantizer.num_embeddings)
    )
    pseudo_sums = pseudo_counts.unsqueeze(1) * centers
    with torch_module.no_grad():
        quantizer.embedding.copy_(
            centers.to(
                device=quantizer.embedding.device,
                dtype=quantizer.embedding.dtype,
            )
        )
        quantizer.ema_cluster_size.copy_(
            pseudo_counts.to(
                device=quantizer.ema_cluster_size.device,
                dtype=quantizer.ema_cluster_size.dtype,
            )
        )
        quantizer.ema_weight.copy_(
            quantizer.ema_cluster_size.unsqueeze(1)
            * quantizer.embedding
        )
    _validate_initialized_ema(quantizer, torch_module)
    return centers, assignments, counts, pseudo_counts, pseudo_sums


def collect_train_prequant_vectors(
    model,
    examples,
    batch_size,
    workers,
    seed,
    device,
    torch_module=torch,
):
    """Collect every fixed, unmasked latent-query projection from train only."""

    if not examples:
        raise VQInitializationError(
            "empty_initialization_partition",
            "training initialization requires training examples",
        )
    captured = []

    def capture(module, inputs, output):
        del module, inputs
        captured.append(output.detach().to(
            device="cpu", dtype=torch_module.float64
        ))

    was_training = model.training
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    handle = model.to_codebook.register_forward_hook(capture)
    model.eval()
    try:
        loader = make_data_loader(
            examples, batch_size, workers, seed, False, torch_module
        )
        with torch_module.no_grad():
            for batch in loader:
                tensors = batch.to_torch(torch_module)
                model.encode_to_memory(
                    tensors["categorical_ids"].to(device),
                    tensors["geometry"].to(device),
                    tensors["geometry_mask"].to(device),
                    tensors["padding_mask"].to(device),
                )
    finally:
        handle.remove()
        model.train(was_training)
    if not captured:
        raise VQInitializationError(
            "missing_prequant_vectors",
            "encoder produced no prequant vectors",
        )
    for name, value in model.state_dict().items():
        if not torch_module.equal(value, state_before[name]):
            raise VQInitializationError(
                "initialization_mutated_model",
                "prequant collection changed model state",
            )
    vectors = torch_module.cat(captured, dim=0).reshape(
        -1, model.vq.embedding_dim
    )
    expected = len(examples) * model.config.latent_tokens
    if vectors.size(0) != expected:
        raise VQInitializationError(
            "invalid_prequant_vector_count",
            "expected exactly one vector per fixed latent-query position",
        )
    if not bool(torch_module.isfinite(vectors).all().item()):
        raise VQInitializationError(
            "nonfinite_prequant",
            "training prequant vectors are non-finite",
        )
    return vectors


def deterministic_kmeans(
    vectors,
    cluster_count,
    seed,
    iterations=KMEANS_LLOYD_ITERATIONS,
    tolerance=DISTINCT_TOLERANCE,
    torch_module=torch,
):
    """Run seeded k-means++ and a fixed number of deterministic Lloyd steps."""

    if (
        vectors.dim() != 2
        or vectors.size(0) < cluster_count
        or cluster_count <= 0
    ):
        raise VQInitializationError(
            "insufficient_prequant_vectors",
            "initialization requires at least one vector per code",
        )
    work = vectors.detach().to(device="cpu", dtype=torch_module.float64)
    if not bool(torch_module.isfinite(work).all().item()):
        raise VQInitializationError(
            "nonfinite_prequant", "prequant vectors are non-finite"
        )
    distinct = _distinct_count(work, tolerance)
    if distinct < cluster_count:
        raise VQInitializationError(
            "insufficient_distinct_prequant_vectors",
            "found {} distinct vectors for {} codes".format(
                distinct, cluster_count
            ),
        )
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    first = int(torch_module.randint(
        work.size(0), (1,), generator=generator
    ).item())
    centers = [work[first].clone()]
    nearest = _squared_distances(work, centers[0].unsqueeze(0)).squeeze(1)
    for _ in range(1, cluster_count):
        total = nearest.sum()
        if not math.isfinite(float(total.item())) or float(total.item()) <= 0:
            raise VQInitializationError(
                "degenerate_kmeans_initialization",
                "k-means++ has no positive selection mass",
            )
        threshold = (
            torch_module.rand((), generator=generator, dtype=work.dtype)
            * total
        )
        cumulative = nearest.cumsum(dim=0)
        index = int(torch_module.searchsorted(
            cumulative, threshold, right=True
        ).item())
        index = min(index, work.size(0) - 1)
        centers.append(work[index].clone())
        candidate = _squared_distances(
            work, centers[-1].unsqueeze(0)
        ).squeeze(1)
        nearest = torch_module.minimum(nearest, candidate)
    centers = torch_module.stack(centers, dim=0)
    assignments = None
    counts = None
    sums = None
    for _ in range(iterations):
        distances = _squared_distances(work, centers)
        # torch.argmin chooses the lowest code index for exact distance ties.
        assignments = distances.argmin(dim=1)
        counts = torch_module.bincount(
            assignments, minlength=cluster_count
        ).to(torch_module.float64)
        sums = torch_module.zeros_like(centers)
        sums.index_add_(0, assignments, work)
        nonempty = counts > 0
        if not bool(nonempty.all().item()):
            raise VQInitializationError(
                "empty_kmeans_cluster",
                "deterministic Lloyd iteration produced an empty cluster",
            )
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    distances = _squared_distances(work, centers)
    assignments = distances.argmin(dim=1)
    counts = torch_module.bincount(
        assignments, minlength=cluster_count
    ).to(torch_module.float64)
    sums = torch_module.zeros_like(centers)
    sums.index_add_(0, assignments, work)
    if bool((counts <= 0).any().item()):
        raise VQInitializationError(
            "empty_kmeans_cluster",
            "final deterministic assignment produced an empty cluster",
        )
    if _distinct_count(centers, tolerance) != cluster_count:
        raise VQInitializationError(
            "indistinct_kmeans_centers",
            "final centers are not distinct at the documented tolerance",
        )
    return centers, assignments, counts, sums


def _validate_initialized_ema(quantizer, torch_module=torch):
    values = (
        quantizer.embedding,
        quantizer.ema_cluster_size,
        quantizer.ema_weight,
    )
    if not all(bool(torch_module.isfinite(value).all().item()) for value in values):
        raise VQInitializationError(
            "inconsistent_ema_state", "initialized EMA state is non-finite"
        )
    if bool((quantizer.ema_cluster_size <= 0).any().item()):
        raise VQInitializationError(
            "inconsistent_ema_state",
            "initialized EMA cluster counts must be positive",
        )
    implied = quantizer.ema_weight / quantizer.ema_cluster_size.unsqueeze(1)
    if not bool(torch_module.allclose(
        implied,
        quantizer.embedding,
        rtol=1e-5,
        atol=1e-6,
    )):
        raise VQInitializationError(
            "inconsistent_ema_state",
            "EMA sums and counts do not reproduce embeddings",
        )


def _squared_distances(vectors, centers):
    return (
        vectors.pow(2).sum(dim=1, keepdim=True)
        + centers.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * torch.matmul(vectors, centers.t())
    ).clamp_min(0.0)


def _distinct_count(vectors, tolerance):
    representatives = []
    for row in vectors:
        if not representatives or all(
            float((row - current).abs().max().item()) > tolerance
            for current in representatives
        ):
            representatives.append(row)
    return len(representatives)


def _tensor_sha256(value):
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    payload = b"".join(
        float(item).hex().encode("ascii") + b"\n"
        for item in tensor.reshape(-1).tolist()
    )
    return hashlib.sha256(payload).hexdigest()


def initialization_report_sha256(report):
    """Hash the canonical report content excluding its self-hash."""

    content = dict(report)
    content.pop("report_sha256", None)
    payload = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()
