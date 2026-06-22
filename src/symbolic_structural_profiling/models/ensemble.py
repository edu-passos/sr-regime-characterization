from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import tempfile
import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..config import EnsembleConfig, FeatureConfig
from ..windowing import FeatureMeta
from .expressions import (
    ExprNode,
    expression_family,
    node_count,
    safe_eval,
    sample_random_expression,
    supported_structure,
)


@lru_cache(maxsize=1)
def _load_pysr_regressor():
    try:
        from pysr import PySRRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency/runtime path
        raise ImportError("PySR is not available") from exc
    return PySRRegressor


@lru_cache(maxsize=1)
def _load_sympy():
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency/runtime path
        raise ImportError("sympy is not available") from exc
    return sp


@dataclass(slots=True)
class SymbolicModel:
    expression: ExprNode
    seed: int
    fit_mse: float
    complexity: int
    equation: str


@dataclass(slots=True)
class WindowEnsembleResult:
    retained_models: List[SymbolicModel]
    candidate_models: List[SymbolicModel]
    baseline_fit_mse: float
    linear_fit_mse: float
    best_symbolic_fit_mse: float
    gate_symbolic_fit_mse: float
    gate_symbolic_per_seed: tuple[float, ...]
    linear_symbolic_fit_gap: float
    seed_instability: float
    backend: str


@dataclass(slots=True)
class EnsembleStageTrace:
    raw_candidates: List[SymbolicModel]
    frontier_retained: List[SymbolicModel]
    retained_pool: List[SymbolicModel]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _is_degenerate_prediction(pred: np.ndarray, min_std: float) -> bool:
    return float(np.std(pred)) < float(min_std)


def persistence_baseline_fit_mse(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    target_index: int,
) -> float:
    lag1_cols = [m.column for m in feature_meta if m.variable == target_index and m.lag == 1]
    if lag1_cols:
        pred = X_fit[:, lag1_cols[0]]
    else:
        pred = np.repeat(np.mean(y_fit), y_fit.shape[0])
    return mse(y_fit, pred)


def _combine_binary(op: str, nodes: list[ExprNode]) -> ExprNode:
    if not nodes:
        return ExprNode(op="const", value=0.0 if op == "add" else 1.0)

    out = nodes[0]
    for node in nodes[1:]:
        out = ExprNode(op=op, children=(out, node))
    return out


def _symbol_to_var_node(symbol_name: str, feature_meta: Sequence[FeatureMeta]) -> ExprNode:
    if not symbol_name.startswith("x"):
        raise ValueError(f"Unsupported symbol: {symbol_name}")

    col = int(symbol_name[1:])
    if col < 0 or col >= len(feature_meta):
        raise ValueError(f"Symbol index out of bounds: {symbol_name}")

    meta = feature_meta[col]
    return ExprNode(op="var", feature_index=meta.column, variable=meta.variable, lag=meta.lag)


def _sympy_to_expr(sym_expr, feature_meta: Sequence[FeatureMeta], sp) -> ExprNode:
    if sym_expr.is_Number:
        return ExprNode(op="const", value=float(sym_expr))

    if sym_expr.is_Symbol:
        return _symbol_to_var_node(str(sym_expr), feature_meta)

    if sym_expr.func == sp.Add:
        args = [_sympy_to_expr(arg, feature_meta, sp) for arg in sym_expr.args]
        return _combine_binary("add", args)

    if sym_expr.func == sp.Mul:
        num_expr, den_expr = sym_expr.as_numer_denom()
        num_terms = [_sympy_to_expr(arg, feature_meta, sp) for arg in sp.Mul.make_args(num_expr)]
        numerator = _combine_binary("mul", num_terms)

        if den_expr == 1:
            return numerator

        den_terms = [_sympy_to_expr(arg, feature_meta, sp) for arg in sp.Mul.make_args(den_expr)]
        denominator = _combine_binary("mul", den_terms)
        return ExprNode(op="div", children=(numerator, denominator))

    if sym_expr.func == sp.Pow:
        base, exponent = sym_expr.args
        if exponent.is_Integer:
            n = int(exponent)
            if n == 0:
                return ExprNode(op="const", value=1.0)
            if n == 1:
                return _sympy_to_expr(base, feature_meta, sp)
            if n == -1:
                return ExprNode(
                    op="div",
                    children=(
                        ExprNode(op="const", value=1.0),
                        _sympy_to_expr(base, feature_meta, sp),
                    ),
                )
            if n > 1:
                factors = [_sympy_to_expr(base, feature_meta, sp) for _ in range(n)]
                return _combine_binary("mul", factors)
        raise ValueError(f"Unsupported power expression: {sym_expr}")

    unary_map = {
        sp.sin: "sin",
        sp.cos: "cos",
        sp.log: "log",
        sp.exp: "exp",
        sp.tanh: "tanh",
    }

    if sym_expr.func in unary_map:
        child = _sympy_to_expr(sym_expr.args[0], feature_meta, sp)
        return ExprNode(op=unary_map[sym_expr.func], children=(child,))

    raise ValueError(f"Unsupported sympy expression: {sym_expr}")


def _equation_string_to_expr(equation: str, feature_meta: Sequence[FeatureMeta]) -> ExprNode:
    sp = _load_sympy()

    locals_map = {
        **{f"x{i}": sp.Symbol(f"x{i}") for i in range(len(feature_meta))},
        "sin": sp.sin,
        "cos": sp.cos,
        "log": sp.log,
        "exp": sp.exp,
        "tanh": sp.tanh,
        "pow": sp.Pow,
    }
    sym_expr = sp.sympify(equation, locals=locals_map)
    return _sympy_to_expr(sym_expr, feature_meta, sp)


def _fit_affine_calibration(pred: np.ndarray, y_fit: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    base_pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(y_fit, dtype=float).reshape(-1)
    if base_pred.shape != target.shape or base_pred.size == 0:
        return None
    if not np.all(np.isfinite(base_pred)) or not np.all(np.isfinite(target)):
        return None

    design = np.column_stack([base_pred, np.ones_like(base_pred)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None

    alpha = float(coeffs[0])
    beta = float(coeffs[1])
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None

    calibrated = alpha * base_pred + beta
    if not np.all(np.isfinite(calibrated)):
        return None
    return alpha, beta, calibrated


def _wrap_affine_expression(expr: ExprNode, alpha: float, beta: float) -> ExprNode:
    if abs(alpha) < 1e-9:
        return ExprNode(op="const", value=float(beta))

    out = expr
    if not np.isclose(alpha, 1.0, atol=1e-6):
        out = ExprNode(
            op="mul",
            children=(
                ExprNode(op="const", value=float(alpha)),
                out,
            ),
        )
    if not np.isclose(beta, 0.0, atol=1e-6):
        out = ExprNode(
            op="add",
            children=(
                out,
                ExprNode(op="const", value=float(beta)),
            ),
        )
    return out


def _random_template_expressions(variable_nodes: Sequence[ExprNode]) -> list[ExprNode]:
    templates: list[ExprNode] = list(variable_nodes)
    for left, right in combinations(variable_nodes, 2):
        templates.append(ExprNode(op="add", children=(left, right)))
        templates.append(ExprNode(op="sub", children=(left, right)))
        templates.append(ExprNode(op="sub", children=(right, left)))
    return templates


def _evaluate_random_candidate(
    expr: ExprNode,
    *,
    seed: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    min_prediction_std: float,
) -> SymbolicModel | None:
    fit_pred = safe_eval(expr, X_fit)
    if _is_degenerate_prediction(fit_pred, min_prediction_std):
        return None

    calibrated = _fit_affine_calibration(fit_pred, y_fit)
    if calibrated is None:
        return None
    alpha, beta, calibrated_pred = calibrated
    if _is_degenerate_prediction(calibrated_pred, min_prediction_std):
        return None

    calibrated_expr = _wrap_affine_expression(expr, alpha, beta)
    fit_err = mse(y_fit, calibrated_pred)
    if not np.isfinite(fit_err):
        return None

    return SymbolicModel(
        expression=calibrated_expr,
        seed=seed,
        fit_mse=fit_err,
        complexity=node_count(calibrated_expr),
        equation=calibrated_expr.to_string(),
    )


def _violates_nested_constraints(
    expr: ExprNode, nested_constraints: dict[str, dict[str, int]]
) -> bool:
    """True if ``expr`` nests an operator inside another against a zero-count
    entry of ``nested_constraints``.

    The random backend accepts ``nested_constraints`` like PySR does, so it must
    honour them rather than silently ignore them. This models the only part of
    PySR's semantics the scaffold is handed in practice: a value of ``0`` in
    ``{outer: {inner: 0}}`` means ``inner`` may not appear anywhere inside
    ``outer``'s subtree (this is exactly nesting level 1 -- no unary-in-unary --
    and level 2 -- also no mul/div in a unary). Nonzero counts (PySR's full
    nesting budget) are deliberately NOT modelled here; if a test needs that, it
    belongs on the PySR backend, not the random scaffold.
    """
    if not nested_constraints:
        return False

    def walk(node: ExprNode, forbidden: frozenset[str]) -> bool:
        if node.op in forbidden:
            return True
        inner = nested_constraints.get(node.op)
        next_forbidden = forbidden
        if inner:
            zeros = frozenset(op for op, count in inner.items() if count == 0)
            if zeros:
                next_forbidden = forbidden | zeros
        return any(walk(child, next_forbidden) for child in node.children)

    return walk(expr, frozenset())


def _fit_random_seed_models(
    seed: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    config: EnsembleConfig,
) -> List[SymbolicModel]:
    binary_op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    variable_nodes = [
        ExprNode(op="var", feature_index=m.column, variable=m.variable, lag=m.lag)
        for m in feature_meta
    ]
    binary_ops = tuple(binary_op_map.get(op, op) for op in config.pysr_binary_operators)
    unary_ops = tuple(config.pysr_unary_operators)

    rng = np.random.default_rng(seed)
    nested_constraints = config.pysr_nested_constraints or {}
    best_by_equation: dict[str, SymbolicModel] = {}

    def _record(expr: ExprNode) -> None:
        if nested_constraints and _violates_nested_constraints(expr, nested_constraints):
            return
        model = _evaluate_random_candidate(
            expr,
            seed=seed,
            X_fit=X_fit,
            y_fit=y_fit,
            min_prediction_std=float(config.min_prediction_std),
        )
        if model is None:
            return
        existing = best_by_equation.get(model.equation)
        if existing is None or model.fit_mse < existing.fit_mse:
            best_by_equation[model.equation] = model

    for expr in _random_template_expressions(variable_nodes):
        _record(expr)

    for _ in range(config.n_candidates_per_seed):
        expr = sample_random_expression(
            rng,
            variable_nodes,
            config.max_expression_depth,
            binary_operators=binary_ops,
            unary_operators=unary_ops,
        )
        _record(expr)

    return sorted(best_by_equation.values(), key=lambda m: (m.fit_mse, m.complexity, m.equation))


def _fit_pysr_seed_models(
    seed: int,
    X_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    feature_meta: Sequence[FeatureMeta] | None = None,
    config: EnsembleConfig | None = None,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> List[SymbolicModel]:
    if X_fit is None:
        X_fit = X_train
    if y_fit is None:
        y_fit = y_train
    if X_fit is None or y_fit is None:
        raise TypeError("_fit_pysr_seed_models requires X_fit/y_fit or X_train/y_train")
    if feature_meta is None or config is None:
        raise TypeError("_fit_pysr_seed_models requires feature_meta and config")

    PySRRegressor = _load_pysr_regressor()
    if int(config.max_evals_per_candidate) <= 0:
        raise ValueError("max_evals_per_candidate must be positive")

    def _coerce_prediction(raw_pred, n_expected: int) -> np.ndarray | None:
        arr = np.asarray(raw_pred, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n_expected, float(arr), dtype=float)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)

        if arr.shape[0] != n_expected:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    def _predict_candidate(idx: int, X: np.ndarray, lambda_fn) -> np.ndarray | None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                try:
                    raw_pred = estimator.predict(X, index=int(idx))
                except Exception:
                    if lambda_fn is None:
                        return None
                    try:
                        raw_pred = lambda_fn(X)
                    except Exception:
                        return None
        return _coerce_prediction(raw_pred, X.shape[0])

    tournament_n = max(2, min(config.pysr_tournament_selection_n, config.pysr_population_size - 1))
    # Force PySR artifacts into an isolated temp directory per fit.
    # This prevents hall_of_fame/checkpoint files from polluting project output folders.
    with tempfile.TemporaryDirectory(prefix="ssp_pysr_") as tempdir:
        estimator = PySRRegressor(
            model_selection="best",
            niterations=config.pysr_niterations,
            populations=config.pysr_populations,
            population_size=config.pysr_population_size,
            maxsize=config.pysr_maxsize,
            maxdepth=config.pysr_maxdepth,
            binary_operators=list(config.pysr_binary_operators),
            unary_operators=list(config.pysr_unary_operators),
            parsimony=config.pysr_parsimony,
            complexity_of_constants=config.pysr_complexity_of_constants,
            complexity_of_operators=config.pysr_complexity_of_operators,
            nested_constraints=config.pysr_nested_constraints or {},
            should_optimize_constants=config.pysr_should_optimize_constants,
            weight_mutate_constant=config.pysr_weight_mutate_constant,
            optimize_probability=config.pysr_optimize_probability,
            ncycles_per_iteration=config.pysr_ncycles_per_iteration,
            tournament_selection_n=tournament_n,
            random_state=seed,
            deterministic=config.pysr_deterministic,
            parallelism=config.pysr_parallelism,
            procs=None if config.pysr_parallelism == "serial" else config.pysr_procs,
            # Bump-allocator backend: pure runtime speedup, no effect on the
            # search or the equations found (allocation strategy only, results
            # are bit-identical). Not a search hyperparameter.
            bumper=True,
            progress=config.pysr_progress,
            verbosity=config.pysr_verbosity,
            temp_equation_file=True,
            tempdir=tempdir,
            delete_tempfiles=True,
        )

        estimator.fit(X_fit, y_fit)

        equations = getattr(estimator, "equations_", None)
        if equations is None or len(equations) == 0:
            return [], float("nan")
        equations = equations.copy()
        if "loss" in equations.columns:
            equations["loss"] = pd.to_numeric(equations["loss"], errors="coerce")
            equations = equations.sort_values("loss", na_position="last")
        equations = equations.head(int(config.max_evals_per_candidate))

        best_by_equation: dict[str, SymbolicModel] = {}

        for idx, row in equations.iterrows():
            equation = str(row.get("equation", ""))
            if not equation:
                continue

            try:
                expr = _equation_string_to_expr(equation, feature_meta)
            except Exception:
                continue

            lambda_fn = row.get("lambda_format")
            fit_pred = _predict_candidate(int(idx), X_fit, lambda_fn)
            if fit_pred is None:
                continue
            if _is_degenerate_prediction(fit_pred, config.min_prediction_std):
                continue

            # Selection error: out-of-sample on the held-out tail when provided,
            # else in-sample. Storing it as fit_mse makes the Pareto frontier and
            # top-K selection downstream automatically reject overfit expressions
            # (they do not improve held-out error).
            if X_val is not None and y_val is not None and X_val.shape[0] > 0:
                val_pred = _predict_candidate(int(idx), X_val, lambda_fn)
                if val_pred is None:
                    continue
                sel_err = mse(y_val, val_pred)
            else:
                sel_err = mse(y_fit, fit_pred)
            if not np.isfinite(sel_err):
                continue

            model = SymbolicModel(
                expression=expr,
                seed=seed,
                fit_mse=sel_err,
                complexity=node_count(expr),
                equation=equation,
            )

            existing = best_by_equation.get(equation)
            if existing is None or model.fit_mse < existing.fit_mse:
                best_by_equation[equation] = model

        # Gate reference (decoupled from tail selection to avoid double-dipping):
        # PySR's own train-selected expression (model_selection="best", chosen from
        # the training rows only -- NOT the held-out tail), scored on the tail. This
        # is the symmetric counterpart to the held-out linear baseline -- both are
        # fit on train and scored on the same tail, neither is selected on it. Taking
        # the tail-selected min instead (best_by_equation) would bias sr_gain upward
        # on exactly the near-zero-gain windows (memory, scaling) the gate must keep
        # silent, because that min is taken over candidates chosen for low tail error.
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            gate_X, gate_y = X_val, y_val
        else:
            gate_X, gate_y = X_fit, y_fit
        native_oos = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                try:
                    native_raw = estimator.predict(gate_X)
                except Exception:
                    native_raw = None
        if native_raw is not None:
            native_pred = _coerce_prediction(native_raw, gate_X.shape[0])
            if native_pred is not None:
                err = mse(gate_y, native_pred)
                if np.isfinite(err):
                    native_oos = float(err)

        return (
            sorted(best_by_equation.values(), key=lambda m: (m.fit_mse, m.complexity, m.equation)),
            native_oos,
        )


def _seed_models_for_backend(
    backend: str,
    seed: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    config: EnsembleConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[List[SymbolicModel], float]:
    if backend == "pysr":
        return _fit_pysr_seed_models(
            seed=seed,
            X_fit=X_fit,
            y_fit=y_fit,
            feature_meta=feature_meta,
            config=config,
            X_val=X_val,
            y_val=y_val,
        )

    if backend == "random":
        # No native train-selected winner for the random backend (fallback only,
        # used when PySR is unavailable); the gate falls back to best_symbolic.
        return (
            _fit_random_seed_models(
                seed=seed,
                X_fit=X_fit,
                y_fit=y_fit,
                feature_meta=feature_meta,
                config=config,
            ),
            float("nan"),
        )

    raise ValueError(f"Unsupported symbolic backend: {backend}")


def _pareto_frontier(models: Sequence[SymbolicModel]) -> List[SymbolicModel]:
    frontier: List[SymbolicModel] = []
    best_fit_so_far = float("inf")

    for model in sorted(models, key=lambda m: (m.complexity, m.fit_mse, m.equation)):
        if model.fit_mse < best_fit_so_far - 1e-12:
            frontier.append(model)
            best_fit_so_far = model.fit_mse

    return frontier


def _select_retained_frontier_models(
    frontier: Sequence[SymbolicModel],
    config: EnsembleConfig,
    *,
    X_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    linear_fit_mse: float | None = None,
    feature_config: FeatureConfig | None = None,
) -> List[SymbolicModel]:
    ordered = sorted(frontier, key=lambda m: (m.fit_mse, m.complexity, m.equation))
    limit = max(1, int(config.top_k_per_seed))
    if not ordered:
        return []
    if not bool(config.rebalance_retained_model_families):
        return ordered[:limit]

    selected: List[SymbolicModel] = []
    seen_equations: set[str] = set()
    feature_cfg = feature_config if feature_config is not None else FeatureConfig()
    family_by_equation: dict[str, str] = {}
    for model in ordered:
        if (
            X_fit is not None
            and y_fit is not None
            and linear_fit_mse is not None
            and np.isfinite(float(linear_fit_mse))
        ):
            family_by_equation[model.equation] = supported_structure(
                model.expression,
                X=X_fit,
                y=y_fit,
                linear_fit_mse=float(linear_fit_mse),
                enabled=bool(feature_cfg.counterfactual_structure_evidence),
                unary_min_ratio=float(feature_cfg.unary_structure_evidence_min_ratio),
                unary_min_abs=float(feature_cfg.unary_structure_evidence_min_abs),
                interaction_min_ratio=float(feature_cfg.interaction_structure_evidence_min_ratio),
                interaction_min_abs=float(feature_cfg.interaction_structure_evidence_min_abs),
            ).family
        else:
            family_by_equation[model.equation] = expression_family(model.expression)

    def _add(model: SymbolicModel | None) -> None:
        if model is None or model.equation in seen_equations or len(selected) >= limit:
            return
        selected.append(model)
        seen_equations.add(model.equation)

    best_overall = ordered[0]
    _add(best_overall)
    represented_families = {family_by_equation[best_overall.equation]}

    best_simple = next(
        (model for model in ordered if family_by_equation[model.equation] == "simple"),
        None,
    )
    if (
        best_simple is not None
        and float(best_simple.fit_mse)
        <= float(best_overall.fit_mse) * float(config.retained_simple_model_fit_ratio) + 1e-12
    ):
        _add(best_simple)
        represented_families.add("simple")

    for family_name in ("unary", "interaction"):
        if family_name in represented_families:
            continue
        family_model = next(
            (model for model in ordered if family_by_equation[model.equation] == family_name),
            None,
        )
        if family_model is not None:
            _add(family_model)
            represented_families.add(family_name)
            break

    for model in ordered:
        _add(model)
        if len(selected) >= limit:
            break

    return selected


def _train_val_split(
    n_rows: int, val_split: float, scheme: str = "tail"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Indices for an out-of-sample split of a window's design rows.

    Returns ``(train_idx, val_idx)`` or ``None`` when no split should apply. The
    held-out rows are used to *select* among PySR's frontier expressions by
    generalization error, which rejects expressions that only lower in-sample
    loss by fitting window noise. The ``"tail"`` scheme holds out the last rows.
    """
    if val_split <= 0.0 or val_split >= 1.0:
        return None
    n_val = max(1, min(int(round(n_rows * val_split)), n_rows - 2))
    n_train = n_rows - n_val
    if n_train < 2:
        return None
    idx = np.arange(n_rows)
    return idx[:n_train], idx[n_train:]  # "tail": hold out the last rows


def _fit_symbolic_ensemble_internal(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    target_index: int,
    config: EnsembleConfig,
    feature_config: FeatureConfig | None = None,
) -> tuple[WindowEnsembleResult, EnsembleStageTrace]:
    # Optional out-of-sample (held-out tail) split. When active, PySR is fit on
    # the train rows and every frontier expression is *selected* by its error on
    # the held-out rows, so noise-fitting complexity is rejected and the
    # symbolic-vs-linear gain is measured out-of-sample. val_split == 0 keeps the
    # original in-sample behaviour exactly.
    val_split = float(getattr(config, "validation_split", 0.0) or 0.0)
    split = _train_val_split(
        int(X_fit.shape[0]), val_split, str(getattr(config, "validation_scheme", "tail"))
    )
    if split is None:
        X_tr, y_tr, X_va, y_va = X_fit, y_fit, None, None
    else:
        tr_idx, va_idx = split
        X_tr, y_tr = X_fit[tr_idx], y_fit[tr_idx]
        X_va, y_va = X_fit[va_idx], y_fit[va_idx]

    baseline_fit_mse = persistence_baseline_fit_mse(X_fit, y_fit, feature_meta, target_index)
    linear_model = Ridge(alpha=1.0)
    linear_model.fit(X_tr, y_tr)
    if X_va is None:
        linear_fit_mse = mse(y_fit, np.asarray(linear_model.predict(X_fit), dtype=float))
    else:
        # held-out linear reference, consistent with the held-out model selection
        linear_fit_mse = mse(y_va, np.asarray(linear_model.predict(X_va), dtype=float))

    requested_backend = config.symbolic_backend
    backend = requested_backend

    if requested_backend == "pysr":
        try:
            _load_pysr_regressor()
        except ImportError:
            backend = config.fallback_backend

    candidates: List[SymbolicModel] = []
    frontier_retained: List[SymbolicModel] = []
    retained: List[SymbolicModel] = []
    best_per_seed: List[float] = []
    native_oos_per_seed: List[float] = []

    for seed in range(config.seed_offset, config.seed_offset + config.n_seeds):
        try:
            seed_models, native_oos = _seed_models_for_backend(
                backend=backend,
                seed=seed,
                X_fit=X_tr,
                y_fit=y_tr,
                feature_meta=feature_meta,
                config=config,
                X_val=X_va,
                y_val=y_va,
            )
        except Exception:
            if backend != config.fallback_backend:
                backend = config.fallback_backend
                seed_models, native_oos = _seed_models_for_backend(
                    backend=backend,
                    seed=seed,
                    X_fit=X_tr,
                    y_fit=y_tr,
                    feature_meta=feature_meta,
                    config=config,
                    X_val=X_va,
                    y_val=y_va,
                )
            else:
                seed_models, native_oos = [], float("nan")

        if not seed_models:
            continue
        if np.isfinite(native_oos):
            native_oos_per_seed.append(float(native_oos))

        candidates.extend(seed_models)
        frontier = _pareto_frontier(seed_models)
        frontier_retained.extend(frontier)
        retained.extend(
            _select_retained_frontier_models(
                frontier,
                config,
                X_fit=X_fit,
                y_fit=y_fit,
                linear_fit_mse=float(linear_fit_mse),
                feature_config=feature_config,
            )
        )
        best_per_seed.append(min(model.fit_mse for model in frontier))

    if best_per_seed:
        seed_instability = float(np.std(best_per_seed) / max(np.median(best_per_seed), 1e-9))
    else:
        seed_instability = 1.0

    candidates.sort(key=lambda m: (m.fit_mse, m.complexity, m.equation))
    frontier_retained.sort(key=lambda m: (m.fit_mse, m.complexity, m.equation))
    retained.sort(key=lambda m: (m.fit_mse, m.complexity, m.equation))

    if retained:
        best_symbolic_fit_mse = retained[0].fit_mse
    else:
        best_symbolic_fit_mse = float("nan")
    linear_symbolic_fit_gap = (
        linear_fit_mse - best_symbolic_fit_mse
        if np.isfinite(best_symbolic_fit_mse)
        else float("nan")
    )

    # Gate's symbolic reference: median over seeds of PySR's train-selected winner
    # scored on the held-out tail (unbiased OOS, decoupled from the tail selection
    # that produces best_symbolic_fit_mse). The median is the ensemble
    # central estimate; a min over seeds would reintroduce tail-selection optimism.
    # With no held-out split (or only the random fallback ran), fall back to the
    # in-sample best so val_split == 0 reproduces the previous behaviour exactly.
    if split is not None and native_oos_per_seed:
        gate_symbolic_fit_mse = float(np.median(native_oos_per_seed))
    else:
        gate_symbolic_fit_mse = best_symbolic_fit_mse

    result = WindowEnsembleResult(
        retained_models=retained,
        candidate_models=candidates,
        baseline_fit_mse=baseline_fit_mse,
        linear_fit_mse=linear_fit_mse,
        best_symbolic_fit_mse=best_symbolic_fit_mse,
        gate_symbolic_fit_mse=gate_symbolic_fit_mse,
        gate_symbolic_per_seed=tuple(native_oos_per_seed),
        linear_symbolic_fit_gap=linear_symbolic_fit_gap,
        seed_instability=seed_instability,
        backend=backend,
    )
    trace = EnsembleStageTrace(
        raw_candidates=list(candidates),
        frontier_retained=list(frontier_retained),
        retained_pool=list(retained),
    )
    return result, trace


def fit_symbolic_ensemble(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    target_index: int,
    config: EnsembleConfig,
    feature_config: FeatureConfig | None = None,
) -> WindowEnsembleResult:
    result, _ = _fit_symbolic_ensemble_internal(
        X_fit=X_fit,
        y_fit=y_fit,
        feature_meta=feature_meta,
        target_index=target_index,
        config=config,
        feature_config=feature_config,
    )
    return result
