"""PTGP: class-balanced task probes and one leading gradient direction."""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def preserved_model_state(model):
    """Keep probe forwards from modifying BatchNorm buffers or train/eval flags."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for module, was_training in modes:
            module.train(was_training)


def make_balanced_groups(dataset, *, per_class: int, groups: int = 32,
                         seed: int = 1234, exclude_indices=()):
    """Sample equal images per class and interleave classes across groups."""
    if per_class < 1 or groups < 1:
        raise ValueError("per_class and groups must be positive")
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "samples", None)
        if labels is not None:
            labels = [label for _, label in labels]
    if labels is None:
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
    labels = np.asarray(labels)
    excluded = set(map(int, exclude_indices))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    by_class = []
    for label in classes:
        eligible = [int(index) for index in np.flatnonzero(labels == label)
                    if int(index) not in excluded]
        if len(eligible) < per_class:
            raise ValueError(f"Class {label} has {len(eligible)} eligible images; need {per_class}")
        by_class.append(rng.choice(eligible, per_class, replace=False).tolist())
    chosen = [by_class[class_index][sample_index]
              for sample_index in range(per_class)
              for class_index in rng.permutation(len(classes))]
    result = [part.tolist() for part in np.array_split(chosen, groups) if len(part)]
    assert len(set(chosen)) == len(chosen)
    return result


class PTGP:
    """Project a parameter gradient onto the leading task-gradient direction.

    A fitted basis has exactly one row. The reference dataset and groups are
    fixed for an experiment; the basis may be recomputed as the model evolves.
    No quantization, additional random directions, or norm clipping is used.
    """

    def __init__(self, reference_dataset, groups, *, microbatch_size=32,
                 probe_mode="eval"):
        self.dataset = reference_dataset
        self.groups = [list(map(int, group)) for group in groups]
        self.microbatch_size = microbatch_size
        if probe_mode not in {"eval", "train"}:
            raise ValueError("probe_mode must be 'eval' or 'train'")
        self.probe_mode = probe_mode
        self.basis = None
        if not self.groups or any(not group for group in self.groups):
            raise ValueError("Reference groups must all be nonempty")

    def fit(self, model):
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("Model has no trainable parameters")
        device = params[0].device
        # Keep the full G matrix on CPU to avoid parameter_count x group_count
        # persistent GPU storage. Each microbatch is evaluated on model device.
        rows = torch.empty((len(self.groups), sum(p.numel() for p in params)),
                           dtype=torch.float32)
        with preserved_model_state(model):
            model.train(self.probe_mode == "train")
            for row_index, group in enumerate(self.groups):
                total = [torch.zeros_like(param) for param in params]
                for offset in range(0, len(group), self.microbatch_size):
                    items = [self.dataset[index]
                             for index in group[offset:offset + self.microbatch_size]]
                    images = torch.stack([image for image, _ in items]).to(device)
                    labels = torch.as_tensor([int(label) for _, label in items], device=device)
                    loss = F.cross_entropy(model(images), labels, reduction="sum") / len(group)
                    gradients = torch.autograd.grad(loss, params)
                    for destination, source in zip(total, gradients):
                        destination.add_(source.detach())
                rows[row_index] = torch.cat([gradient.reshape(-1) for gradient in total]).cpu()
        if not torch.isfinite(rows).all():
            raise FloatingPointError("Non-finite reference gradient")
        gram = rows.double() @ rows.double().T
        if hasattr(torch, "linalg"):
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        else:
            eigenvalues, eigenvectors = torch.symeig(gram, eigenvectors=True)
        if eigenvalues[-1] <= 0 or not torch.isfinite(eigenvalues[-1]):
            raise FloatingPointError("Invalid principal task-gradient energy")
        leading = eigenvectors[:, -1].float() @ rows
        leading = leading / leading.norm()
        self.basis = leading.to(device)
        return {"reference_groups": len(self.groups),
                "leading_eigenvalue": float(eigenvalues[-1]),
                "retained_directions": 1}

    def project(self, gradients):
        if self.basis is None:
            raise RuntimeError("Call fit(model) before project(gradients)")
        flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        if flat.numel() != self.basis.numel():
            raise ValueError("Gradient does not match the fitted model")
        projected = torch.dot(flat, self.basis) * self.basis
        parts = projected.split([gradient.numel() for gradient in gradients])
        transformed = tuple(part.reshape_as(gradient) for part, gradient in zip(parts, gradients))
        retained = float((projected * projected).sum() /
                         (flat * flat).sum().clamp_min(1e-30))
        return transformed, {"retained_energy_fraction": retained}
