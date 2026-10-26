"""
Defense methods.

Including:
- Additive noise
- Gradient clipping
- Gradient compression
- Representation perturbation
"""

# The code of defense is adapted from https://github.com/zhuohangli/GGL. 

import numpy as np
from gpu_math import percentile_linear

import torch
import torch.nn as nn
import torch.optim as optim
import math
import logging
import os
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(123)


def _ginner(left, right):
    """Compute and process model gradients."""
    
    return sum((x * y).sum() for x, y in zip(left, right))


def _gadd(left, right, alpha=1.0):
    """Compute and process model gradients."""
    
    return [x + alpha * y for x, y in zip(left, right)]


def _gproject(gradient, basis):
    """Compute an orthogonal gradient-space direction."""
    
    result = [torch.zeros_like(part) for part in gradient]
    
    for unit in basis:
        
        result = _gadd(result, unit, alpha=_ginner(gradient, unit))
    
    return result


def _gorthonormalize(directions, eps=1e-12):
    """Compute modified Gram--Schmidt orthonormalization in gradient space."""
    
    basis = []
    
    for direction in directions:
        
        residual = [part.detach().clone() for part in direction]
        
        for unit in basis:
            residual = _gadd(residual, unit, alpha=-_ginner(residual, unit))
        
        norm = torch.sqrt(_ginner(residual, residual).clamp_min(eps))
        
        if torch.isfinite(norm) and float(norm) > eps:
            
            basis.append([part / norm for part in residual])
    
    return basis


def _gradient_at_input(model, inputs, labels, criterion):
    """Compute and process model gradients."""
    
    model.zero_grad()
    
    loss = criterion(model(inputs), labels)
    
    gradient = torch.autograd.grad(loss, tuple(model.parameters()))
    
    return [part.detach() for part in gradient]


def _virtual_gmm_loss(model, gradient, inputs, labels, learning_rate, criterion):
    """Evaluate one local virtual update and restore model state exactly."""
    parameters = list(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    was_training = model.training
    try:
        with torch.no_grad():
            for parameter, update in zip(parameters, gradient):
                parameter.copy_(parameter - float(learning_rate) * update)
        model.eval()
        with torch.no_grad():
            return float(criterion(model(inputs), labels).item())
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
        model.train(was_training)


def _mean_classification_loss(model, views, labels, criterion):
    """Evaluate classification loss and preserve model state."""
    
    return float(sum(criterion(model(view), labels) for view in views).item() / len(views))


def _virtual_mean_loss(model, gradient, views, labels, learning_rate, criterion):
    """Experiment support utility."""
    parameters = list(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    was_training = model.training
    try:
        with torch.no_grad():
            for parameter, update in zip(parameters, gradient):
                parameter.copy_(parameter - float(learning_rate) * update)
        model.eval()
        with torch.no_grad():
            return _mean_classification_loss(model, views, labels, criterion)
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
        model.train(was_training)


def _build_label_preserving_views(inputs, count, noise_std, generator):
    """Experiment support utility."""
    views = [inputs.detach()]
    for index in range(1, max(int(count), 1)):
        view = inputs.detach()
        if index % 2 == 1:
            view = torch.flip(view, dims=(-1,))
        shift = (index % 3) - 1
        if shift:
            view = torch.roll(view, shifts=(shift, -shift), dims=(-2, -1))
        if float(noise_std) > 0:
            noise = torch.randn(view.shape, dtype=view.dtype,
                                device=view.device, generator=generator)
            view = view + float(noise_std) * noise
        views.append(view)
    return views


def _estimate_gradient_manifold(model, ground_truth, labels, criterion,
                                tangent_rank, fd_epsilon, seed,
                                base_gradient=None, include_curvature=False,
                                fd_scales=(1.0,)):
    """Compute and process model gradients."""
    
    tangents = []
    
    devices = [ground_truth.device] if ground_truth.is_cuda else []
    
    with torch.random.fork_rng(devices=devices):
        
        torch.manual_seed(int(seed))
        
        for _ in range(max(int(tangent_rank), 0)):
            
            input_direction = torch.randn_like(ground_truth)
            
            input_direction = input_direction / input_direction.square().mean().sqrt().clamp_min(1e-12)
            for scale in fd_scales:
                epsilon = float(fd_epsilon) * float(scale)
                plus = ground_truth.detach() + epsilon * input_direction
                minus = ground_truth.detach() - epsilon * input_direction
                grad_plus = _gradient_at_input(model, plus, labels, criterion)
                grad_minus = _gradient_at_input(model, minus, labels, criterion)
                
                tangents.append([(a - b) / (2.0 * epsilon)
                                 for a, b in zip(grad_plus, grad_minus)])
                if include_curvature and base_gradient is not None:
                    
                    tangents.append([(a + b - 2.0 * g) / (epsilon * epsilon)
                                     for a, b, g in zip(
                                         grad_plus, grad_minus, base_gradient)])
    
    return _gorthonormalize(tangents)


def _remove_reversible_directions(original_gradient, manifold_basis):
    """Experiment support utility."""
    
    reversible = _gproject(original_gradient, manifold_basis)
    
    filtered = _gadd(original_gradient, reversible, alpha=-1.0)
    
    return filtered, reversible


def _project_safe_descent_region(original, reversible, manifold_basis,
                                 model, ground_truth, labels, criterion,
                                 virtual_lr, min_descent_cosine,
                                 utility_tolerance, norm_scales,
                                 contraction_steps,
                                 max_residual_tangent_fraction,
                                 bicone_enabled=False,
                                 bicone_safe_cosine=0.70,
                                 bicone_candidates=16,
                                 bicone_temperature=0.20,
                                 matchability_weight=0.0,
                                 seed=1234):
    """Project gradients onto the selected subspace."""
    
    original_norm = torch.sqrt(_ginner(original, original).clamp_min(1e-12))
    
    current_loss = float(criterion(model(ground_truth), labels).item())
    
    chosen, fallback, fallback_key = None, None, None
    evaluated_count, total_feasible_count = 0, 0
    
    generator = torch.Generator(device=original[0].device)
    generator.manual_seed(int(seed))
    
    for alpha_tensor in torch.linspace(1.0, 0.0, max(int(contraction_steps), 1) + 1):
        layer_chosen, layer_gumbel_score, layer_feasible_count = None, -float('inf'), 0
        
        alpha = float(alpha_tensor.item())
        
        direction = _gadd(original, reversible, alpha=-alpha)
        
        direction_norm = torch.sqrt(_ginner(direction, direction).clamp_min(1e-12))
        
        safe_unit = [part / direction_norm for part in direction]
        
        direction_count = max(int(bicone_candidates), 1) if bicone_enabled else 1
        safe_cosine = min(max(float(bicone_safe_cosine), 0.0), 1.0)
        max_angle = math.acos(safe_cosine)
        for direction_index in range(direction_count):
            if direction_index == 0:
                direction_unit = safe_unit
            else:
                noise = [torch.randn(part.shape, dtype=part.dtype,
                                     device=part.device, generator=generator)
                         for part in original]
                
                noise = _gadd(noise, _gproject(noise, manifold_basis), alpha=-1.0)
                noise = _gadd(noise, safe_unit, alpha=-_ginner(noise, safe_unit))
                noise_norm = torch.sqrt(_ginner(noise, noise).clamp_min(1e-12))
                noise_unit = [part / noise_norm for part in noise]
                
                
                angle = max_angle * float(direction_index) / float(max(direction_count - 1, 1))
                direction_unit = [math.cos(angle) * axis + math.sin(angle) * ortho
                                  for axis, ortho in zip(safe_unit, noise_unit)]
            
            for scale in norm_scales:
                
                candidate = [part * (float(scale) * original_norm) for part in direction_unit]
                candidate_norm = torch.sqrt(_ginner(candidate, candidate).clamp_min(1e-12))
                cosine = float((_ginner(candidate, original) /
                                (candidate_norm * original_norm)).item())
                safe_cosine_value = float((_ginner(candidate, safe_unit) /
                                           candidate_norm).item())
                residual = _gproject(candidate, manifold_basis)
                residual_norm = torch.sqrt(_ginner(residual, residual).clamp_min(0.0))
                residual_fraction = float((residual_norm / candidate_norm).item())
                virtual_loss = _virtual_gmm_loss(
                    model, candidate, ground_truth, labels, virtual_lr, criterion)
                record = (alpha, float(scale), candidate, cosine,
                          residual_fraction, virtual_loss, safe_cosine_value)
                evaluated_count += 1
                
                candidate_key = (
                    max(0.0, virtual_loss - current_loss - float(utility_tolerance)),
                    max(0.0, float(min_descent_cosine) - cosine),
                    residual_fraction, -alpha)
                if fallback_key is None or candidate_key < fallback_key:
                    fallback, fallback_key = record, candidate_key
                if (cosine >= float(min_descent_cosine)
                        and (not bicone_enabled or
                             safe_cosine_value >= float(bicone_safe_cosine) - 1e-6)
                        and residual_fraction <= float(max_residual_tangent_fraction)
                        and virtual_loss <= current_loss + float(utility_tolerance)):
                    layer_feasible_count += 1
                    total_feasible_count += 1
                    if not bicone_enabled:
                        layer_chosen = record
                        break
                    
                    
                    energy = (residual_fraction
                              + float(matchability_weight) * abs(cosine)
                              + max(0.0, virtual_loss - current_loss))
                    uniform = float(torch.rand((), generator=generator,
                                                 device=original[0].device).clamp_(1e-7, 1 - 1e-7).item())
                    gumbel = -math.log(-math.log(uniform))
                    score = -energy / max(float(bicone_temperature), 1e-6) + gumbel
                    if score > layer_gumbel_score:
                        layer_chosen, layer_gumbel_score = record, score
                    
                    break
            if layer_chosen is not None and not bicone_enabled:
                break
        
        if layer_chosen is not None:
            chosen = layer_chosen
            break
    
    feasible = chosen is not None
    
    if chosen is None:
        chosen = fallback
    
    return chosen, feasible, evaluated_count, current_loss, total_feasible_count


def gmm_gradient(input_gradient, model, ground_truth, labels,
                 tangent_rank=4, fd_epsilon=0.03, virtual_lr=1e-4,
                 min_descent_cosine=0.05, utility_tolerance=0.02,
                 norm_scales=(1.0, 0.75, 0.5, 0.25), seed=1234,
                 contraction_steps=20, max_residual_tangent_fraction=1.0,
                 bicone_enabled=False, bicone_safe_cosine=0.70,
                 bicone_candidates=16, bicone_temperature=0.20,
                 robust_reversible_space=False, fd_scales=(1.0,),
                 matchability_weight=0.0):
    """Compute and process model gradients."""
    
    criterion = nn.CrossEntropyLoss()
    
    original = [part.detach().clone() for part in input_gradient]
    
    original_norm = torch.sqrt(_ginner(original, original).clamp_min(1e-12))
    
    was_training = model.training
    
    model.eval()

    
    manifold_basis = _estimate_gradient_manifold(
        model, ground_truth, labels, criterion,
        tangent_rank, fd_epsilon, seed, base_gradient=original,
        include_curvature=robust_reversible_space, fd_scales=fd_scales)
    
    filtered, reversible = _remove_reversible_directions(
        original, manifold_basis)
    
    chosen, feasible, evaluated_count, current_loss, feasible_count = _project_safe_descent_region(
        original, reversible, manifold_basis, model, ground_truth, labels,
        criterion, virtual_lr, min_descent_cosine, utility_tolerance,
        norm_scales, contraction_steps, max_residual_tangent_fraction,
        bicone_enabled, bicone_safe_cosine, bicone_candidates,
        bicone_temperature, matchability_weight, seed)

    
    selected_alpha, selected_scale, protected, alignment, residual_fraction, protected_loss, safe_alignment = chosen
    
    filtered_norm = torch.sqrt(_ginner(filtered, filtered).clamp_min(1e-12))
    
    reversible_norm = torch.sqrt(_ginner(reversible, reversible).clamp_min(0.0))
    
    diagnostics = {
        'method_version': ('gradient_manifold_margin_bicone_v2'
                           if robust_reversible_space else
                           ('gradient_manifold_margin_bicone' if bicone_enabled
                            else 'gradient_manifold_margin_original_three_stage')),
        'tangent_rank_requested': int(tangent_rank),
        'tangent_rank_effective': len(manifold_basis),
        'fd_epsilon': float(fd_epsilon),
        'reversible_energy_fraction': float(reversible_norm.item() / max(float(original_norm.item()), 1e-12)),
        'filtered_energy_fraction': float(filtered_norm.item() / max(float(original_norm.item()), 1e-12)),
        'tangent_contraction_alpha': float(selected_alpha),
        'residual_tangent_fraction': float(residual_fraction),
        'descent_cosine': float(alignment),
        'safe_axis_cosine': float(safe_alignment),
        'bicone_enabled': int(bool(bicone_enabled)),
        'robust_reversible_space': int(bool(robust_reversible_space)),
        'matchability_weight': float(matchability_weight),
        'bicone_feasible_count': int(feasible_count),
        'selected_norm_scale': float(selected_scale),
        'current_loss': float(current_loss),
        'protected_virtual_loss': float(protected_loss),
        'utility_constraint_met': int(feasible),
        'candidate_count_evaluated': int(evaluated_count),
    }
    
    logger.info('GMM diagnostics: %s', diagnostics)
    
    model.train(was_training)
    
    model.zero_grad()
    
    return [part.detach().clone() for part in protected], diagnostics


def _tensor_orthogonal_noise(reference, basis_parts, generator, eps=1e-7):
    """Compute an orthogonal gradient-space direction."""
    
    noise = torch.randn(reference.shape, dtype=reference.dtype,
                        device=reference.device, generator=generator)
    
    directions = _gorthonormalize(
        [[reference]] + [[part] for part in basis_parts if part is not None], eps=eps)
    
    for unit_tuple in directions:
        
        direction = unit_tuple[0]
        
        noise = noise - (noise * direction).sum() * direction
    
    norm = noise.square().sum().sqrt()
    
    if float(norm.item()) <= float(eps) * max(math.sqrt(reference.numel()), 1.0):
        
        return torch.zeros_like(reference)
    
    unit = noise / norm
    
    reference_cosine = abs(float(
        ((unit * reference).sum() / reference.square().sum().sqrt().clamp_min(eps)).item()))
    
    if reference_cosine > 1e-4:
        return torch.zeros_like(reference)
    
    return unit




def gmm_budget_gradient(input_gradient, model, ground_truth, labels,
                                  tangent_rank=2, fd_epsilon=0.03,
                                  fd_scales=(0.5, 1.5), virtual_lr=1e-4,
                                  utility_budget=0.02, risk_utility_lambda=1.0,
                                  boundary_min_alpha=0.80, candidate_count=4,
                                  utility_view_count=3, utility_noise_std=0.01,
                                  norm_jitter=0.10, seed=1234,
                                  min_descent_retention=0.70,
                                  relative_utility_budget=0.10):
    """Compute an orthogonal gradient-space direction."""
    
    criterion = nn.CrossEntropyLoss()
    
    original = [part.detach().clone() for part in input_gradient]
    
    was_training = model.training
    
    model.eval()
    
    generator = torch.Generator(device=original[0].device)
    
    generator.manual_seed(int(seed))

    
    views = _build_label_preserving_views(
        ground_truth, utility_view_count, utility_noise_std, generator)
    
    view_gradients = [_gradient_at_input(model, view, labels, criterion)
                      for view in views]
    
    task_anchor = [sum(parts) / len(parts) for parts in zip(*view_gradients)]
    
    basis = _estimate_gradient_manifold(
        model, ground_truth, labels, criterion, tangent_rank,
        fd_epsilon, seed, base_gradient=original,
        include_curvature=True, fd_scales=fd_scales)
    
    reversible = _gproject(original, basis)
    
    task_safe = _gadd(task_anchor, _gproject(task_anchor, basis), alpha=-1.0)

    
    names = [name for name, _ in model.named_parameters()]
    
    group_map = {}
    
    for index, name in enumerate(names):
        
        group_map.setdefault(name.rsplit('.', 1)[0], []).append(index)
    
    groups = list(group_map.values())
    
    total_task_energy = max(sum(part.square().sum().item()
                                for part in task_anchor), 1e-12)
    
    risks, utilities = [], []
    
    for indices in groups:
        
        leak_energy = sum(reversible[i].square().sum() for i in indices)
        
        grad_energy = sum(original[i].square().sum() for i in indices).clamp_min(1e-12)
        
        risks.append(float((leak_energy / grad_energy).sqrt().item()))
        
        utilities.append(float(sum(task_anchor[i].square().sum().item()
                                   for i in indices) / total_task_energy))
    
    max_risk, max_utility = max(risks, default=1.0), max(utilities, default=1.0)
    
    risk_norm = [value / max(max_risk, 1e-12) for value in risks]
    
    utility_norm = [value / max(max_utility, 1e-12) for value in utilities]
    
    group_alphas = [risk / (risk + float(risk_utility_lambda) * utility + 1e-12)
                    for risk, utility in zip(risk_norm, utility_norm)]
    
    if groups:
        
        group_alphas[0] = max(group_alphas[0], float(boundary_min_alpha))
        
        group_alphas[-1] = max(group_alphas[-1], float(boundary_min_alpha))
    
    tensor_alphas = [0.0] * len(original)
    
    for group_index, indices in enumerate(groups):
        
        for index in indices:
            tensor_alphas[index] = group_alphas[group_index]

    
    baseline_loss = _virtual_mean_loss(
        model, original, views, labels, virtual_lr, criterion)
    
    current_loss = _mean_classification_loss(model, views, labels, criterion)
    
    
    original_descent = max(float(current_loss - baseline_loss), 0.0)
    required_descent = float(min_descent_retention) * original_descent
    
    allowed_gap = max(float(utility_budget),
                      float(relative_utility_budget) * original_descent)
    
    gamma_schedule = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.0)
    
    chosen, chosen_key, feasible_count, evaluated_count = None, None, 0, 0
    
    selected_gamma = 0.0
    
    for gamma in gamma_schedule:
        
        level_best, level_key, level_feasible = None, None, 0
        
        for _ in range(max(int(candidate_count), 1)):
            
            candidate, matches = [], []
            
            for index, (grad, safe_part, anchor_part) in enumerate(
                    zip(original, task_safe, task_anchor)):
                
                alpha = float(gamma) * float(tensor_alphas[index])
                
                if alpha > 0:
                    
                    unit = _tensor_orthogonal_noise(
                        grad, [unit_basis[index] for unit_basis in basis], generator)
                    
                    jitter = 2.0 * float(torch.rand((), generator=generator,
                                                     device=grad.device).item()) - 1.0
                    
                    private_part = (unit * anchor_part.square().sum().sqrt().clamp_min(1e-12)
                                    * math.exp(float(norm_jitter) * jitter))
                    
                    protected_part = (1.0 - alpha) * safe_part + alpha * private_part
                
                else:
                    protected_part = safe_part
                
                candidate.append(protected_part)
                
                denominator = (protected_part.square().sum().sqrt()
                               * grad.square().sum().sqrt()).clamp_min(1e-12)
                
                matches.append(abs(float(((protected_part * grad).sum() /
                                          denominator).item())))
            
            virtual_loss = _virtual_mean_loss(
                model, candidate, views, labels, virtual_lr, criterion)
            
            evaluated_count += 1
            
            candidate_descent = float(current_loss - virtual_loss)
            descent_violation = max(0.0, required_descent - candidate_descent)
            baseline_violation = max(0.0, virtual_loss - baseline_loss - allowed_gap)
            violation = max(descent_violation, baseline_violation)
            
            weighted_match = sum(risk * max(matches[index] for index in indices)
                                 for risk, indices in zip(risk_norm, groups)) / max(sum(risk_norm), 1e-12)
            
            key = (violation, weighted_match, virtual_loss)
            
            if violation == 0.0:
                
                level_feasible += 1
                
                feasible_count += 1
            
            if level_key is None or key < level_key:
                level_best, level_key = (candidate, virtual_loss, matches), key
        
        if level_feasible:
            
            chosen, chosen_key, selected_gamma = level_best, level_key, float(gamma)
            
            break
        
        if chosen_key is None or level_key < chosen_key:
            chosen, chosen_key, selected_gamma = level_best, level_key, float(gamma)

    
    
    utility_repair_alpha = 1.0
    if chosen_key is None or chosen_key[0] > 0.0:
        private_candidate = chosen[0]
        for private_alpha in (0.875, 0.75, 0.625, 0.50, 0.375, 0.25,
                              0.125, 0.0):
            repaired = [(1.0 - private_alpha) * raw + private_alpha * private
                        for raw, private in zip(original, private_candidate)]
            repaired_loss = _virtual_mean_loss(
                model, repaired, views, labels, virtual_lr, criterion)
            repaired_descent = float(current_loss - repaired_loss)
            descent_violation = max(0.0, required_descent - repaired_descent)
            baseline_violation = max(
                0.0, repaired_loss - baseline_loss - allowed_gap)
            violation = max(descent_violation, baseline_violation)
            repaired_matches = []
            for raw, part in zip(original, repaired):
                denominator = (raw.square().sum().sqrt()
                               * part.square().sum().sqrt()).clamp_min(1e-12)
                repaired_matches.append(abs(float(
                    ((raw * part).sum() / denominator).item())))
            weighted_match = sum(
                risk * max(repaired_matches[index] for index in indices)
                for risk, indices in zip(risk_norm, groups)) / max(
                    sum(risk_norm), 1e-12)
            if violation == 0.0:
                chosen = (repaired, repaired_loss, repaired_matches)
                chosen_key = (0.0, weighted_match, repaired_loss)
                utility_repair_alpha = float(private_alpha)
                feasible_count += 1
                break

    
    protected, protected_loss, matches = chosen
    
    diagnostics = {
        'method_version': 'gmm_budget_risk_utility_budget_v2',
        'module_group_count': len(groups),
        'tangent_rank_effective': len(basis),
        'mean_group_risk': float(np.mean(risks)),
        'max_group_risk': float(np.max(risks)),
        'mean_alpha': float(np.mean(group_alphas)),
        'max_alpha': float(np.max(group_alphas)),
        'selected_privacy_gamma': selected_gamma,
        'utility_repair_private_alpha': utility_repair_alpha,
        'current_mean_loss': current_loss,
        'baseline_update_mean_loss': baseline_loss,
        'protected_virtual_loss': float(protected_loss),
        'utility_gap_vs_original_update': float(protected_loss - baseline_loss),
        'original_descent': original_descent,
        'required_descent': required_descent,
        'protected_descent': float(current_loss - protected_loss),
        'descent_retention': float(
            (current_loss - protected_loss) / max(original_descent, 1e-12))
            if original_descent > 0 else 1.0,
        'min_descent_retention': float(min_descent_retention),
        'relative_utility_budget': float(relative_utility_budget),
        'utility_constraint_met': int(chosen_key[0] == 0.0),
        'feasible_candidate_count': feasible_count,
        'candidate_count_evaluated': evaluated_count,
        'weighted_matchability': float(chosen_key[1]),
    }
    
    logger.info('GMM risk-utility budget diagnostics: %s', diagnostics)
    
    model.train(was_training)
    
    model.zero_grad()
    
    return [part.detach().clone() for part in protected], diagnostics

# def additive_noise(input_gradient, std=0.1):
def additive_noise(model, input_gradient, save_dir, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0 # np.inf
    device = input_gradient[0].device
    grad_tensor = [g.detach() for g in input_gradient]
    
    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grad_tensor]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    gradient = []
    for grad in input_gradient:
        thresh = percentile_linear(grad.abs(), percentage)
        gradient.append(torch.where(grad.abs() < thresh, 0, grad))
    return gradient


def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device
    
    gt_data = ground_truth.clone()
    gt_data.requires_grad=True

    # register forward hook to get intermediate layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]
        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)
    
    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:,f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/((feature_graph.data[:,f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:,f] = 0
        
    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = percentile_linear(deviation_x_norm_sum, pruning_rate)
    mask = torch.where(deviation_x_norm_sum.abs() < thresh, 0., 1.).to(dtype=input_gradient[-2].dtype)
    
    print('Soteria mask: ', mask.sum().item())

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * mask
    
    handle.remove()
    
    return gradient

def generate_orthogonal_gradient(input_gradient):
    """
    Generate a new gradient that is orthogonal to the input_gradient.

    Args:
    - input_gradient: A list of gradients (tensors) for each layer of the model.

    Returns:
    - orthogonal_gradient: A list of gradients (tensors) that are orthogonal to the input_gradient.
    """
    orthogonal_gradient = []

    for grad in input_gradient:
        # Generate a random gradient with the same shape
        random_grad = torch.randn_like(grad)
        random_grad = random_grad / torch.norm(random_grad)

        # Flatten the gradients to treat them as vectors
        grad_flat = grad.flatten()
        random_grad_flat = random_grad.flatten()

        # Compute the projection of random_grad onto grad
        proj_scalar = torch.dot(random_grad_flat, grad_flat) / torch.norm(grad_flat)
        proj_vector = proj_scalar * grad_flat

        # Subtract the projection from random_grad to make it orthogonal
        orthogonal_grad_flat = random_grad_flat - proj_vector
        
        # Reshape back to the original shape and append to the list
        orthogonal_grad = orthogonal_grad_flat.view_as(grad)
        orthogonal_gradient.append(orthogonal_grad)

    return orthogonal_gradient

def normalize_orthogonal_gradient(ortho_gradient, original_gradient, fixed_norm=None):
    normalized_gradient = []
    for og, orig_g in zip(ortho_gradient, original_gradient):
        norm_og = torch.norm(og.flatten())
        norm_orig_g = fixed_norm if fixed_norm is not None else torch.norm(orig_g.flatten())
        normalized_g = (og / norm_og) * norm_orig_g if norm_og > 0 else og
        normalized_gradient.append(normalized_g)
    return normalized_gradient


def orthogonal_gradient(input_gradient, model, ground_truth, labels, trials, epsilon=1e-5, best_loss=float('inf')):
    criterion = nn.CrossEntropyLoss()
    best_gradient = None
    learning_rate = 1e-4

    original_params = [param.clone() for param in model.parameters()]  # Save original parameters outside the loop

    for trial in range(trials):
        model.train()
        noisy_gradient = generate_orthogonal_gradient(input_gradient)
        noisy_gradient = normalize_orthogonal_gradient(noisy_gradient, input_gradient)
        
        with torch.no_grad():
            for param, original, noise_grad in zip(model.parameters(), original_params, noisy_gradient):
                param.data.copy_(original - noise_grad * learning_rate)  # Update and evaluate in one step
            
            model.eval()
            new_loss = criterion(model(ground_truth), labels)

            # Logging less frequently
            if trial % 5 == 0:
                logger.info(f"{trial} trial loss: {new_loss}")

            # Revert to original parameters if needed
            if new_loss < best_loss:
                best_loss = new_loss
                best_gradient = [g.detach().clone() for g in noisy_gradient]  # Only clone when necessary
                logger.info(f'Gradient updated, Loss reduced to {best_loss}')

    # Restore original parameters
    for param, original in zip(model.parameters(), original_params):
        param.data.copy_(original)

    if best_gradient is not None:
        input_gradient = [g.clone() for g in best_gradient]

    return input_gradient, best_loss
