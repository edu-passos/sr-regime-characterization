from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence

import numpy as np


BINARY_OPS = ("add", "sub", "mul", "div")
UNARY_OPS = ("sin", "cos", "tanh", "exp", "log", "neg")
UNARY_NONLINEAR_OPS = {"sin", "cos", "tanh", "exp", "log"}


@dataclass(slots=True)
class ExprNode:
    op: str
    children: tuple["ExprNode", ...] = ()
    feature_index: int | None = None
    variable: int | None = None
    lag: int | None = None
    value: float | None = None

    def to_string(self) -> str:
        if self.op == "var":
            return f"x{self.variable}_lag{self.lag}"
        if self.op == "const":
            return f"{self.value:.4f}"
        if self.op in BINARY_OPS:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
            return f"({left} {symbol} {right})"
        if self.op == "neg":
            return f"(-{self.children[0].to_string()})"
        return f"{self.op}({self.children[0].to_string()})"


@dataclass(slots=True)
class SupportedStructure:
    core_expr: ExprNode
    supported_expr: ExprNode
    raw_has_nonlinearity: bool
    raw_has_unary_nonlinearity: bool
    raw_has_interaction: bool
    raw_has_coupled_interaction: bool
    raw_is_simple_model: bool
    raw_family: str
    has_nonlinearity: bool
    has_unary_nonlinearity: bool
    has_interaction: bool
    has_coupled_interaction: bool
    is_simple_model: bool
    family: str
    full_fit_mse: float
    simple_surrogate_fit_mse: float
    single_lag_unary_surrogate_fit_mse: float
    additive_noninteraction_surrogate_fit_mse: float
    unary_pruned_fit_mse: float
    interaction_pruned_fit_mse: float
    unary_curvature_gain: float
    interaction_synergy_gain: float
    unary_evidence_gain: float
    interaction_evidence_gain: float


@dataclass(slots=True)
class SupportedStructureAudit:
    core_expr: ExprNode
    single_lag_unary_expr: ExprNode
    additive_noninteraction_expr: ExprNode
    simple_surrogate_expr: ExprNode
    supported_expr_before_final_strip: ExprNode
    supported_expr_after_final_strip: ExprNode
    full_fit_mse: float
    single_lag_unary_surrogate_fit_mse: float
    additive_noninteraction_surrogate_fit_mse: float
    simple_surrogate_fit_mse: float
    unary_curvature_gain: float
    interaction_synergy_gain: float
    unary_abs_threshold: float
    unary_ratio_threshold: float
    unary_effective_threshold: float
    interaction_abs_threshold: float
    interaction_ratio_threshold: float
    interaction_effective_threshold: float
    unary_surrogate_has_unary: bool
    interaction_supported: bool
    unary_supported: bool
    original_unary_effect_gain: float
    surrogate_unary_effect_gain_raw: float
    surrogate_unary_effect_gain: float
    unary_preservation_ratio: float
    unary_support_path: str
    strict_failed_only_due_to_surrogate_unary_absence: bool
    supported_family_before_final_strip: str
    supported_family_after_final_strip: str
    collapsed_to_simple_before_final_strip: bool
    collapsed_to_simple_after_final_strip: bool
    unary_failure_reason: str


def safe_eval(node: ExprNode, X: np.ndarray) -> np.ndarray:
    if node.op == "var":
        return X[:, node.feature_index]
    if node.op == "const":
        return np.full(X.shape[0], float(node.value), dtype=float)

    if node.op in BINARY_OPS:
        a = safe_eval(node.children[0], X)
        b = safe_eval(node.children[1], X)
        if node.op == "add":
            out = a + b
        elif node.op == "sub":
            out = a - b
        elif node.op == "mul":
            out = a * b
        else:
            eps = 1e-6
            # Keep denominator away from zero while preserving sign.
            sign = np.where(b < 0.0, -1.0, 1.0)
            denom = np.where(np.abs(b) < eps, sign * eps, b)
            out = a / denom
    else:
        a = safe_eval(node.children[0], X)
        if node.op == "sin":
            out = np.sin(a)
        elif node.op == "cos":
            out = np.cos(a)
        elif node.op == "tanh":
            out = np.tanh(a)
        elif node.op == "exp":
            out = np.exp(np.clip(a, -8.0, 8.0))
        elif node.op == "log":
            out = np.log(np.abs(a) + 1e-6)
        elif node.op == "neg":
            out = -a
        else:
            raise ValueError(f"Unsupported operation: {node.op}")

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def iter_nodes(node: ExprNode) -> Iterable[ExprNode]:
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def _is_constant_subtree(node: ExprNode) -> bool:
    if node.op == "const":
        return True
    if node.op == "var":
        return False
    return all(_is_constant_subtree(child) for child in node.children)


def _constant_subtree_value(node: ExprNode) -> float | None:
    if not _is_constant_subtree(node):
        return None
    value = float(safe_eval(node, np.zeros((1, 1), dtype=float))[0])
    return value if np.isfinite(value) else None


def strip_affine_envelope(node: ExprNode) -> ExprNode:
    current = node
    while True:
        if current.op == "neg":
            current = current.children[0]
            continue

        if current.op == "add":
            left, right = current.children
            if _is_constant_subtree(left) and not _is_constant_subtree(right):
                current = right
                continue
            if _is_constant_subtree(right) and not _is_constant_subtree(left):
                current = left
                continue

        if current.op == "sub":
            left, right = current.children
            if _is_constant_subtree(right) and not _is_constant_subtree(left):
                current = left
                continue
            if _is_constant_subtree(left) and not _is_constant_subtree(right):
                current = right
                continue

        if current.op == "mul":
            left, right = current.children
            left_value = _constant_subtree_value(left)
            right_value = _constant_subtree_value(right)
            if left_value is not None and abs(left_value) > 1e-12 and not _is_constant_subtree(right):
                current = right
                continue
            if right_value is not None and abs(right_value) > 1e-12 and not _is_constant_subtree(left):
                current = left
                continue

        if current.op == "div":
            left, right = current.children
            right_value = _constant_subtree_value(right)
            if right_value is not None and abs(right_value) > 1e-12 and not _is_constant_subtree(left):
                current = left
                continue

        return current


def _is_coupled_interaction_carrier(node: ExprNode) -> bool:
    return (
        node.op in UNARY_NONLINEAR_OPS
        and len(_lag_reference_set(node.children[0])) > 1
    )


def strip_pure_unary_nonlinearity(node: ExprNode) -> ExprNode:
    if node.op in {"var", "const"}:
        return node
    if node.op == "neg":
        child = strip_pure_unary_nonlinearity(node.children[0])
        return ExprNode(op="neg", children=(child,))
    if node.op in UNARY_NONLINEAR_OPS:
        child = strip_pure_unary_nonlinearity(node.children[0])
        if _is_coupled_interaction_carrier(node):
            return ExprNode(op=node.op, children=(child,))
        return child
    if node.op in BINARY_OPS:
        left = strip_pure_unary_nonlinearity(node.children[0])
        right = strip_pure_unary_nonlinearity(node.children[1])
        return ExprNode(op=node.op, children=(left, right))
    child = strip_pure_unary_nonlinearity(node.children[0])
    return ExprNode(op=node.op, children=(child,))


def _family_from_flags(*, has_unary_nonlinearity: bool, has_interaction: bool) -> str:
    if has_interaction:
        return "interaction"
    if has_unary_nonlinearity:
        return "unary"
    return "simple"


def _family_from_expr(node: ExprNode) -> str:
    has_unary = bool(has_unary_nonlinearity(node))
    interaction_count, _, _ = interaction_stats(node)
    has_interaction_flag = bool(interaction_count > 0)
    return _family_from_flags(
        has_unary_nonlinearity=has_unary,
        has_interaction=has_interaction_flag,
    )


def _unary_failure_reason(
    *,
    interaction_supported: bool,
    unary_surrogate_has_unary: bool,
    unary_gain: float,
    unary_abs_threshold: float,
    unary_ratio_threshold: float,
    unary_effective_threshold: float,
    supported_family_before_final_strip: str,
    supported_family_after_final_strip: str,
) -> str:
    if interaction_supported:
        return "interaction_supported_dominates"
    if (
        supported_family_before_final_strip == "unary"
        and supported_family_after_final_strip == "simple"
    ):
        return "collapsed_to_simple_after_affine_strip"
    if not unary_surrogate_has_unary:
        return "unary_surrogate_lacks_unary"

    below_abs = bool(unary_gain < unary_abs_threshold - 1e-12)
    below_ratio = bool(unary_gain < unary_ratio_threshold - 1e-12)
    below_effective = bool(unary_gain < unary_effective_threshold - 1e-12)

    if below_abs and below_ratio:
        return "gain_below_joint_threshold"
    if below_abs:
        return "gain_below_abs_threshold"
    if below_ratio:
        return "gain_below_ratio_threshold"
    if below_effective:
        return "gain_below_joint_threshold"
    return "other"


def _unary_preservation_ratio(
    *,
    original_unary_effect_gain: float,
    surrogate_unary_effect_gain_raw: float,
) -> tuple[float, float]:
    surrogate_unary_effect_gain = max(float(surrogate_unary_effect_gain_raw), 0.0)
    if not np.isfinite(float(original_unary_effect_gain)) or float(original_unary_effect_gain) <= 0.0:
        return surrogate_unary_effect_gain, float("nan")
    return surrogate_unary_effect_gain, float(
        surrogate_unary_effect_gain / float(original_unary_effect_gain)
    )


def _structure_evidence_threshold(
    linear_fit_mse: float,
    *,
    min_ratio: float,
    min_abs: float,
) -> float:
    if not np.isfinite(linear_fit_mse):
        return float(min_abs)
    return max(float(linear_fit_mse) * float(min_ratio), float(min_abs))


def affine_calibrated_mse(node: ExprNode, X: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(safe_eval(node, X), dtype=float).reshape(-1)
    target = np.asarray(y, dtype=float).reshape(-1)
    if pred.shape != target.shape or pred.size == 0:
        return float("inf")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        return float("inf")

    design = np.column_stack([pred, np.ones_like(pred)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    alpha = float(coeffs[0])
    beta = float(coeffs[1])
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return float("inf")

    calibrated = alpha * pred + beta
    if not np.all(np.isfinite(calibrated)):
        return float("inf")
    return float(np.mean((target - calibrated) ** 2))


def _deterministic_branch_choice(
    left: ExprNode,
    right: ExprNode,
    X: np.ndarray,
    y: np.ndarray,
) -> ExprNode:
    left_score = affine_calibrated_mse(left, X, y)
    right_score = affine_calibrated_mse(right, X, y)
    left_key = (float(left_score), node_count(left), left.to_string())
    right_key = (float(right_score), node_count(right), right.to_string())
    return left if left_key <= right_key else right


def _single_lag_descendants(node: ExprNode) -> list[ExprNode]:
    candidates: list[ExprNode] = []
    seen: set[str] = set()
    for candidate in iter_nodes(node):
        refs = _lag_reference_set(candidate)
        if len(refs) != 1:
            continue
        key = candidate.to_string()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _deterministic_expr_choice(
    candidates: Sequence[ExprNode],
    X: np.ndarray,
    y: np.ndarray,
) -> ExprNode | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            float(affine_calibrated_mse(candidate, X, y)),
            node_count(candidate),
            candidate.to_string(),
        ),
    )


def _dedupe_expr_candidates(candidates: Sequence[ExprNode]) -> list[ExprNode]:
    out: list[ExprNode] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.to_string()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _additive_surrogate_candidates(candidates: Sequence[ExprNode]) -> list[ExprNode]:
    base = _dedupe_expr_candidates(candidates)
    out: list[ExprNode] = list(base)
    limited = base[:6]
    for left, right in combinations(limited, 2):
        out.append(ExprNode(op="add", children=(left, right)))
        out.append(ExprNode(op="sub", children=(left, right)))
        out.append(ExprNode(op="sub", children=(right, left)))
    return _dedupe_expr_candidates(out)


def _is_explicit_interaction_binary(node: ExprNode) -> bool:
    if node.op not in {"mul", "div"}:
        return False

    left_refs = _lag_reference_set(node.children[0])
    right_refs = _lag_reference_set(node.children[1])
    if not left_refs or not right_refs:
        return False
    return left_refs != right_refs


def single_lag_unary_surrogate(node: ExprNode, X: np.ndarray, y: np.ndarray) -> ExprNode:
    if node.op in {"var", "const"}:
        return node
    if node.op == "neg":
        child = single_lag_unary_surrogate(node.children[0], X, y)
        return ExprNode(op="neg", children=(child,))
    if node.op in BINARY_OPS:
        left = single_lag_unary_surrogate(node.children[0], X, y)
        right = single_lag_unary_surrogate(node.children[1], X, y)
        candidate = ExprNode(op=node.op, children=(left, right))
        if len(_lag_reference_set(candidate)) <= 1:
            return candidate
        branch_candidates = _dedupe_expr_candidates([left, right, *_single_lag_descendants(node)])
        best = _deterministic_expr_choice(branch_candidates, X, y)
        return best if best is not None else left
    if node.op in UNARY_NONLINEAR_OPS:
        child = single_lag_unary_surrogate(node.children[0], X, y)
        candidate = ExprNode(op=node.op, children=(child,))
        if len(_lag_reference_set(candidate)) <= 1:
            return candidate
        wrapped_candidates = [
            ExprNode(op=node.op, children=(descendant,))
            for descendant in _single_lag_descendants(node.children[0])
        ]
        best = _deterministic_expr_choice(_dedupe_expr_candidates([candidate, *wrapped_candidates]), X, y)
        return best if best is not None else candidate
    child = single_lag_unary_surrogate(node.children[0], X, y)
    return ExprNode(op=node.op, children=(child,))


def additive_noninteraction_surrogate(node: ExprNode, X: np.ndarray, y: np.ndarray) -> ExprNode:
    if node.op in {"var", "const"}:
        return node
    if node.op == "neg":
        child = additive_noninteraction_surrogate(node.children[0], X, y)
        return ExprNode(op="neg", children=(child,))
    if node.op in {"add", "sub"}:
        left = additive_noninteraction_surrogate(node.children[0], X, y)
        right = additive_noninteraction_surrogate(node.children[1], X, y)
        return ExprNode(op=node.op, children=(left, right))
    if node.op in {"mul", "div"}:
        left = additive_noninteraction_surrogate(node.children[0], X, y)
        right = additive_noninteraction_surrogate(node.children[1], X, y)
        candidate = ExprNode(op=node.op, children=(left, right))
        if not _is_explicit_interaction_binary(candidate):
            return candidate
        additive_candidates = _additive_surrogate_candidates([left, right])
        best = _deterministic_expr_choice(additive_candidates, X, y)
        return best if best is not None else left
    if node.op in UNARY_NONLINEAR_OPS:
        child = additive_noninteraction_surrogate(node.children[0], X, y)
        candidate = ExprNode(op=node.op, children=(child,))
        if len(_lag_reference_set(candidate)) <= 1:
            return candidate
        wrapped_terms = [
            ExprNode(op=node.op, children=(descendant,))
            for descendant in _single_lag_descendants(child)
        ]
        best = _deterministic_expr_choice(_dedupe_expr_candidates(wrapped_terms), X, y)
        return best if best is not None else candidate
    child = additive_noninteraction_surrogate(node.children[0], X, y)
    return ExprNode(op=node.op, children=(child,))


def _supported_structure_impl(
    node: ExprNode,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    linear_fit_mse: float | None = None,
    enabled: bool = True,
    unary_min_ratio: float = 0.05,
    unary_min_abs: float = 1e-4,
    interaction_min_ratio: float = 0.05,
    interaction_min_abs: float = 1e-4,
    unary_support_policy: str = "strict",
    unary_preserve_ratio: float = 0.33,
) -> tuple[SupportedStructure, SupportedStructureAudit]:
    core_expr = strip_affine_envelope(node)
    raw_has_unary = bool(has_unary_nonlinearity(core_expr))
    raw_interaction_count, _, _ = interaction_stats(core_expr)
    raw_has_interaction = bool(raw_interaction_count > 0)
    raw_has_coupled = bool(has_coupled_interaction(core_expr))
    raw_has_nonlinearity = bool(nonlinear_node_count(core_expr) > 0)
    raw_is_simple = (not raw_has_unary) and (not raw_has_interaction)
    raw_family = _family_from_flags(
        has_unary_nonlinearity=raw_has_unary,
        has_interaction=raw_has_interaction,
    )

    if (
        not enabled
        or X is None
        or y is None
        or not np.isfinite(float(linear_fit_mse) if linear_fit_mse is not None else float("nan"))
    ):
        support = SupportedStructure(
            core_expr=core_expr,
            supported_expr=core_expr,
            raw_has_nonlinearity=raw_has_nonlinearity,
            raw_has_unary_nonlinearity=raw_has_unary,
            raw_has_interaction=raw_has_interaction,
            raw_has_coupled_interaction=raw_has_coupled,
            raw_is_simple_model=raw_is_simple,
            raw_family=raw_family,
            has_nonlinearity=raw_has_nonlinearity,
            has_unary_nonlinearity=raw_has_unary,
            has_interaction=raw_has_interaction,
            has_coupled_interaction=raw_has_coupled,
            is_simple_model=raw_is_simple,
            family=raw_family,
            full_fit_mse=float("nan"),
            simple_surrogate_fit_mse=float("nan"),
            single_lag_unary_surrogate_fit_mse=float("nan"),
            additive_noninteraction_surrogate_fit_mse=float("nan"),
            unary_pruned_fit_mse=float("nan"),
            interaction_pruned_fit_mse=float("nan"),
            unary_curvature_gain=float("nan"),
            interaction_synergy_gain=float("nan"),
            unary_evidence_gain=float("nan"),
            interaction_evidence_gain=float("nan"),
        )
        audit = SupportedStructureAudit(
            core_expr=core_expr,
            single_lag_unary_expr=core_expr,
            additive_noninteraction_expr=core_expr,
            simple_surrogate_expr=core_expr,
            supported_expr_before_final_strip=core_expr,
            supported_expr_after_final_strip=core_expr,
            full_fit_mse=float("nan"),
            single_lag_unary_surrogate_fit_mse=float("nan"),
            additive_noninteraction_surrogate_fit_mse=float("nan"),
            simple_surrogate_fit_mse=float("nan"),
            unary_curvature_gain=float("nan"),
            interaction_synergy_gain=float("nan"),
            unary_abs_threshold=float(unary_min_abs),
            unary_ratio_threshold=float("nan"),
            unary_effective_threshold=float(unary_min_abs),
            interaction_abs_threshold=float(interaction_min_abs),
            interaction_ratio_threshold=float("nan"),
            interaction_effective_threshold=float(interaction_min_abs),
            unary_surrogate_has_unary=raw_has_unary,
            interaction_supported=raw_has_interaction,
            unary_supported=raw_has_unary,
            original_unary_effect_gain=float("nan"),
            surrogate_unary_effect_gain_raw=float("nan"),
            surrogate_unary_effect_gain=float("nan"),
            unary_preservation_ratio=float("nan"),
            unary_support_path="strict" if raw_has_unary else ("interaction" if raw_has_interaction else "none"),
            strict_failed_only_due_to_surrogate_unary_absence=False,
            supported_family_before_final_strip=raw_family,
            supported_family_after_final_strip=raw_family,
            collapsed_to_simple_before_final_strip=(raw_family == "simple"),
            collapsed_to_simple_after_final_strip=(raw_family == "simple"),
            unary_failure_reason="",
        )
        return support, audit

    linear_fit_value = float(linear_fit_mse)
    full_fit = affine_calibrated_mse(core_expr, X, y)
    single_lag_unary_expr = single_lag_unary_surrogate(core_expr, X, y)
    additive_noninteraction_expr = additive_noninteraction_surrogate(core_expr, X, y)
    simple_surrogate_expr = strip_affine_envelope(
        strip_pure_unary_nonlinearity(additive_noninteraction_expr)
    )
    single_lag_unary_fit = affine_calibrated_mse(single_lag_unary_expr, X, y)
    additive_noninteraction_fit = affine_calibrated_mse(additive_noninteraction_expr, X, y)
    simple_surrogate_fit = affine_calibrated_mse(simple_surrogate_expr, X, y)
    unary_surrogate_has_unary = bool(has_unary_nonlinearity(single_lag_unary_expr))

    unary_gain = float(simple_surrogate_fit - single_lag_unary_fit)
    interaction_gain = float(additive_noninteraction_fit - full_fit)
    original_unary_effect_gain = float(simple_surrogate_fit - full_fit)
    surrogate_unary_effect_gain_raw = float(simple_surrogate_fit - single_lag_unary_fit)
    surrogate_unary_effect_gain, unary_preservation_ratio = _unary_preservation_ratio(
        original_unary_effect_gain=original_unary_effect_gain,
        surrogate_unary_effect_gain_raw=surrogate_unary_effect_gain_raw,
    )
    unary_abs_threshold = float(unary_min_abs)
    unary_ratio_threshold = (
        float(linear_fit_value) * float(unary_min_ratio)
        if np.isfinite(linear_fit_value)
        else float("nan")
    )
    unary_threshold = _structure_evidence_threshold(
        linear_fit_value,
        min_ratio=float(unary_min_ratio),
        min_abs=float(unary_min_abs),
    )
    interaction_abs_threshold = float(interaction_min_abs)
    interaction_ratio_threshold = (
        float(linear_fit_value) * float(interaction_min_ratio)
        if np.isfinite(linear_fit_value)
        else float("nan")
    )
    interaction_threshold = _structure_evidence_threshold(
        linear_fit_value,
        min_ratio=float(interaction_min_ratio),
        min_abs=float(interaction_min_abs),
    )

    interaction_supported = (
        raw_has_interaction
        and interaction_gain >= interaction_threshold - 1e-12
        and interaction_gain > unary_gain + 1e-12
    )
    strict_unary_supported = (
        unary_surrogate_has_unary
        and not interaction_supported
        and unary_gain >= unary_threshold - 1e-12
    )
    strict_failed_only_due_to_surrogate_unary_absence = bool(
        raw_family == "unary"
        and not interaction_supported
        and not strict_unary_supported
        and not unary_surrogate_has_unary
        and original_unary_effect_gain >= unary_threshold - 1e-12
        and surrogate_unary_effect_gain >= unary_threshold - 1e-12
    )
    preserve_fraction_supported = bool(
        unary_support_policy == "preserve_fraction"
        and raw_family == "unary"
        and not interaction_supported
        and original_unary_effect_gain >= unary_threshold - 1e-12
        and surrogate_unary_effect_gain > 0.0
        and np.isfinite(unary_preservation_ratio)
        and unary_preservation_ratio >= float(unary_preserve_ratio) - 1e-12
    )
    unary_supported = bool(strict_unary_supported or preserve_fraction_supported)
    unary_support_path = "none"

    if interaction_supported:
        unary_support_path = "interaction"
        supported_expr_before_final_strip = core_expr
    elif strict_unary_supported:
        unary_support_path = "strict"
        supported_expr_before_final_strip = single_lag_unary_expr
    elif preserve_fraction_supported:
        unary_support_path = "preserve_fraction"
        supported_expr_before_final_strip = core_expr
    else:
        supported_expr_before_final_strip = simple_surrogate_expr
    supported_expr = strip_affine_envelope(supported_expr_before_final_strip)

    supported_family_before_final_strip = _family_from_expr(
        supported_expr_before_final_strip
    )

    supported_has_unary = bool(has_unary_nonlinearity(supported_expr))
    supported_interaction_count, _, _ = interaction_stats(supported_expr)
    supported_has_interaction = bool(supported_interaction_count > 0)
    supported_has_coupled = bool(has_coupled_interaction(supported_expr))
    supported_has_nonlinearity = bool(nonlinear_node_count(supported_expr) > 0)
    supported_is_simple = (not supported_has_unary) and (not supported_has_interaction)
    supported_family = _family_from_flags(
        has_unary_nonlinearity=supported_has_unary,
        has_interaction=supported_has_interaction,
    )
    unary_failure_reason = ""
    if supported_family != "unary" and (
        raw_family == "unary"
        or unary_surrogate_has_unary
        or supported_family_before_final_strip == "unary"
        or interaction_supported
    ):
        unary_failure_reason = _unary_failure_reason(
            interaction_supported=interaction_supported,
            unary_surrogate_has_unary=unary_surrogate_has_unary,
            unary_gain=unary_gain,
            unary_abs_threshold=float(unary_abs_threshold),
            unary_ratio_threshold=float(unary_ratio_threshold),
            unary_effective_threshold=float(unary_threshold),
            supported_family_before_final_strip=supported_family_before_final_strip,
            supported_family_after_final_strip=supported_family,
        )

    support = SupportedStructure(
        core_expr=core_expr,
        supported_expr=supported_expr,
        raw_has_nonlinearity=raw_has_nonlinearity,
        raw_has_unary_nonlinearity=raw_has_unary,
        raw_has_interaction=raw_has_interaction,
        raw_has_coupled_interaction=raw_has_coupled,
        raw_is_simple_model=raw_is_simple,
        raw_family=raw_family,
        has_nonlinearity=supported_has_nonlinearity,
        has_unary_nonlinearity=supported_has_unary,
        has_interaction=supported_has_interaction,
        has_coupled_interaction=supported_has_coupled,
        is_simple_model=supported_is_simple,
        family=supported_family,
        full_fit_mse=float(full_fit),
        simple_surrogate_fit_mse=float(simple_surrogate_fit),
        single_lag_unary_surrogate_fit_mse=float(single_lag_unary_fit),
        additive_noninteraction_surrogate_fit_mse=float(additive_noninteraction_fit),
        unary_pruned_fit_mse=float(simple_surrogate_fit),
        interaction_pruned_fit_mse=float(additive_noninteraction_fit),
        unary_curvature_gain=float(unary_gain),
        interaction_synergy_gain=float(interaction_gain),
        unary_evidence_gain=float(unary_gain),
        interaction_evidence_gain=float(interaction_gain),
    )
    audit = SupportedStructureAudit(
        core_expr=core_expr,
        single_lag_unary_expr=single_lag_unary_expr,
        additive_noninteraction_expr=additive_noninteraction_expr,
        simple_surrogate_expr=simple_surrogate_expr,
        supported_expr_before_final_strip=supported_expr_before_final_strip,
        supported_expr_after_final_strip=supported_expr,
        full_fit_mse=float(full_fit),
        single_lag_unary_surrogate_fit_mse=float(single_lag_unary_fit),
        additive_noninteraction_surrogate_fit_mse=float(additive_noninteraction_fit),
        simple_surrogate_fit_mse=float(simple_surrogate_fit),
        unary_curvature_gain=float(unary_gain),
        interaction_synergy_gain=float(interaction_gain),
        unary_abs_threshold=float(unary_abs_threshold),
        unary_ratio_threshold=float(unary_ratio_threshold),
        unary_effective_threshold=float(unary_threshold),
        interaction_abs_threshold=float(interaction_abs_threshold),
        interaction_ratio_threshold=float(interaction_ratio_threshold),
        interaction_effective_threshold=float(interaction_threshold),
        unary_surrogate_has_unary=bool(unary_surrogate_has_unary),
        interaction_supported=bool(interaction_supported),
        unary_supported=bool(unary_supported),
        original_unary_effect_gain=float(original_unary_effect_gain),
        surrogate_unary_effect_gain_raw=float(surrogate_unary_effect_gain_raw),
        surrogate_unary_effect_gain=float(surrogate_unary_effect_gain),
        unary_preservation_ratio=float(unary_preservation_ratio),
        unary_support_path=str(unary_support_path),
        strict_failed_only_due_to_surrogate_unary_absence=bool(
            strict_failed_only_due_to_surrogate_unary_absence
        ),
        supported_family_before_final_strip=str(supported_family_before_final_strip),
        supported_family_after_final_strip=str(supported_family),
        collapsed_to_simple_before_final_strip=bool(
            supported_family_before_final_strip == "simple"
        ),
        collapsed_to_simple_after_final_strip=bool(supported_family == "simple"),
        unary_failure_reason=str(unary_failure_reason),
    )
    return support, audit


def supported_structure(
    node: ExprNode,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    linear_fit_mse: float | None = None,
    enabled: bool = True,
    unary_min_ratio: float = 0.05,
    unary_min_abs: float = 1e-4,
    interaction_min_ratio: float = 0.05,
    interaction_min_abs: float = 1e-4,
    unary_support_policy: str = "strict",
    unary_preserve_ratio: float = 0.33,
) -> SupportedStructure:
    support, _ = _supported_structure_impl(
        node,
        X=X,
        y=y,
        linear_fit_mse=linear_fit_mse,
        enabled=enabled,
        unary_min_ratio=unary_min_ratio,
        unary_min_abs=unary_min_abs,
        interaction_min_ratio=interaction_min_ratio,
        interaction_min_abs=interaction_min_abs,
        unary_support_policy=unary_support_policy,
        unary_preserve_ratio=unary_preserve_ratio,
    )
    return support


def audit_supported_structure(
    node: ExprNode,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    linear_fit_mse: float | None = None,
    enabled: bool = True,
    unary_min_ratio: float = 0.05,
    unary_min_abs: float = 1e-4,
    interaction_min_ratio: float = 0.05,
    interaction_min_abs: float = 1e-4,
    unary_support_policy: str = "strict",
    unary_preserve_ratio: float = 0.33,
) -> tuple[SupportedStructure, SupportedStructureAudit]:
    return _supported_structure_impl(
        node,
        X=X,
        y=y,
        linear_fit_mse=linear_fit_mse,
        enabled=enabled,
        unary_min_ratio=unary_min_ratio,
        unary_min_abs=unary_min_abs,
        interaction_min_ratio=interaction_min_ratio,
        interaction_min_abs=interaction_min_abs,
        unary_support_policy=unary_support_policy,
        unary_preserve_ratio=unary_preserve_ratio,
    )


def node_count(node: ExprNode) -> int:
    return sum(1 for _ in iter_nodes(node))


def expression_depth(node: ExprNode) -> int:
    if not node.children:
        return 1
    return 1 + max(expression_depth(c) for c in node.children)


def nonlinear_node_count(node: ExprNode) -> int:
    return sum(1 for n in iter_nodes(node) if _is_structurally_nonlinear(n))


def max_nonlinear_composition_depth(node: ExprNode) -> int:
    def _rec(n: ExprNode, current: int) -> int:
        increment = 1 if _is_structurally_nonlinear(n) else 0
        next_depth = current + increment
        if not n.children:
            return next_depth
        return max(_rec(child, next_depth) for child in n.children)

    return _rec(node, 0)


def _contains_variable(node: ExprNode) -> bool:
    if node.op == "var":
        return True
    return any(_contains_variable(child) for child in node.children)


def _lag_reference_set(node: ExprNode) -> set[tuple[int, int]]:
    core = strip_affine_envelope(node)
    return used_variables(core)


def has_unary_nonlinearity(node: ExprNode) -> bool:
    core = strip_affine_envelope(node)
    return any(
        n.op in UNARY_NONLINEAR_OPS and not _is_coupled_interaction_carrier(n)
        for n in iter_nodes(core)
    )


def has_coupled_interaction(node: ExprNode) -> bool:
    core = strip_affine_envelope(node)
    return any(_is_coupled_interaction_carrier(n) for n in iter_nodes(core))


def expression_family(node: ExprNode) -> str:
    core = strip_affine_envelope(node)
    interaction_count, _, _ = interaction_stats(core)
    if interaction_count > 0:
        return "interaction"
    if has_unary_nonlinearity(core):
        return "unary"
    return "simple"


def _is_valid_interaction_mul(node: ExprNode) -> bool:
    return node.op == "mul" and _is_explicit_interaction_binary(node)


def _is_structurally_nonlinear(node: ExprNode) -> bool:
    if node.op in UNARY_NONLINEAR_OPS:
        if _is_coupled_interaction_carrier(node):
            return False
        return True
    if node.op == "mul":
        return (
            not _is_valid_interaction_mul(node)
            and _contains_variable(node.children[0])
            and _contains_variable(node.children[1])
        )
    if node.op == "div":
        if _is_explicit_interaction_binary(node):
            return False
        return bool(_lag_reference_set(node.children[1]))
    return False


def polynomial_degree_proxy(node: ExprNode) -> int:
    if node.op == "const":
        return 0
    if node.op == "var":
        return 1
    if node.op in {"add", "sub"}:
        return max(polynomial_degree_proxy(node.children[0]), polynomial_degree_proxy(node.children[1]))
    if node.op == "mul":
        return polynomial_degree_proxy(node.children[0]) + polynomial_degree_proxy(node.children[1])
    if node.op == "div":
        return max(1, polynomial_degree_proxy(node.children[0]) + polynomial_degree_proxy(node.children[1]))
    if node.op == "neg":
        return polynomial_degree_proxy(node.children[0])
    return max(2, polynomial_degree_proxy(node.children[0]))


def variable_occurrences(node: ExprNode) -> List[tuple[int, int, int]]:
    out: List[tuple[int, int, int]] = []
    for n in iter_nodes(node):
        if n.op == "var" and n.feature_index is not None:
            out.append((n.feature_index, n.variable, n.lag))
    return out


def used_variables(node: ExprNode) -> set[tuple[int, int]]:
    return {(v, l) for _, v, l in variable_occurrences(node)}


def max_lag_used(node: ExprNode) -> int:
    used = variable_occurrences(node)
    if not used:
        return 0
    return int(max(lag for _, _, lag in used))


def _interaction_from_binary(node: ExprNode) -> tuple[int, int] | None:
    if not _is_explicit_interaction_binary(node):
        return None

    left_vars = _lag_reference_set(node.children[0])
    right_vars = _lag_reference_set(node.children[1])
    merged = left_vars | right_vars
    return len(merged), expression_depth(node)


def _interaction_from_coupled_unary(node: ExprNode) -> tuple[int, int] | None:
    if not _is_coupled_interaction_carrier(node):
        return None
    return len(_lag_reference_set(node.children[0])), expression_depth(node)


def interaction_stats(node: ExprNode) -> tuple[int, int, int]:
    orders: List[int] = []
    depths: List[int] = []
    count = 0

    def _visit(current: ExprNode) -> None:
        nonlocal count
        coupled_result = _interaction_from_coupled_unary(current)
        if coupled_result is not None:
            order, depth = coupled_result
            count += 1
            orders.append(order)
            depths.append(depth)
            return

        binary_result = _interaction_from_binary(current)
        if binary_result is not None:
            order, depth = binary_result
            count += 1
            orders.append(order)
            depths.append(depth)

        for child in current.children:
            _visit(child)

    _visit(node)

    if count == 0:
        return 0, 1, 0
    return count, int(max(orders)), int(max(depths))


def operator_histogram(node: ExprNode) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for n in iter_nodes(node):
        hist[n.op] = hist.get(n.op, 0) + 1
    return hist


def finite_difference_importance(
    node: ExprNode,
    X: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    baseline = safe_eval(node, X)
    n_features = X.shape[1]
    out = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        scale = np.std(X[:, j])
        step = epsilon * (scale if scale > 1e-9 else 1.0)
        X2 = X.copy()
        X2[:, j] += step
        pred = safe_eval(node, X2)
        out[j] = float(np.mean(np.abs(pred - baseline)) / max(step, 1e-12))

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def sample_random_expression(
    rng: np.random.Generator,
    variable_nodes: Sequence[ExprNode],
    max_depth: int,
    binary_operators: Sequence[str] | None = None,
    unary_operators: Sequence[str] | None = None,
) -> ExprNode:
    binary_ops = tuple(binary_operators or BINARY_OPS)
    unary_ops = tuple(unary_operators or UNARY_OPS)

    if max_depth <= 0:
        if rng.random() < 0.82:
            return variable_nodes[int(rng.integers(0, len(variable_nodes)))]
        return ExprNode(op="const", value=float(rng.normal(0.0, 1.0)))

    draw = rng.random()
    if draw < 0.30:
        return variable_nodes[int(rng.integers(0, len(variable_nodes)))]
    if draw < 0.42:
        return ExprNode(op="const", value=float(rng.normal(0.0, 1.0)))

    if binary_ops and (not unary_ops or rng.random() < 0.58):
        op = binary_ops[int(rng.integers(0, len(binary_ops)))]
        left = sample_random_expression(
            rng,
            variable_nodes,
            max_depth - 1,
            binary_operators=binary_ops,
            unary_operators=unary_ops,
        )
        right = sample_random_expression(
            rng,
            variable_nodes,
            max_depth - 1,
            binary_operators=binary_ops,
            unary_operators=unary_ops,
        )
        return ExprNode(op=op, children=(left, right))

    if not unary_ops:
        return variable_nodes[int(rng.integers(0, len(variable_nodes)))]

    op = unary_ops[int(rng.integers(0, len(unary_ops)))]
    child = sample_random_expression(
        rng,
        variable_nodes,
        max_depth - 1,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
    )
    return ExprNode(op=op, children=(child,))
