# Main script to run train + evaluation

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils.evaluation_utils import ResultsManager, EvaluationResult
from utils.data_utils import load_config, prepare_data
from utils.tracking_utils import get_tracker, safe_log


def _make_run_name(config: dict, clf_type: str, pipeline: str) -> str:
    """
    Create an intuitive run name so it's easy to identify in the Trackio dashboard.

    Example:
      Month_2__grid_search_with_final_eval__logreg__rs42-123
    """
    dataset_path = config.get("data", {}).get("dataset_path", "dataset")
    dataset_stem = Path(dataset_path).stem if dataset_path else "dataset"

    # NOTE: random_state values are config (reproducibility), not metrics.
    return f"{dataset_stem}__{pipeline}__{clf_type}"


def _make_group_name(config: dict, pipeline: str) -> str:
    """
    Group runs so teammates can filter by dataset + pipeline.
    """
    dataset_path = config.get("data", {}).get("dataset_path", "dataset")
    dataset_stem = Path(dataset_path).stem if dataset_path else "dataset"
    return f"{dataset_stem}__{pipeline}"


def _run_metadata(config: dict, clf_type: str, pipeline: str) -> Dict[str, Any]:
    """Structured metadata for collaboration/filtering."""
    dataset_path = str(config.get("data", {}).get("dataset_path", ""))
    dataset_stem = Path(dataset_path).stem if dataset_path else "dataset"

    # good-enough “who ran it” fields
    user = (
        os.getenv("TRACKIO_USER")
        or os.getenv("GITHUB_ACTOR")
        or os.getenv("USER")
        or os.getenv("USERNAME")
        or "unknown"
    )
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"

    # IMPORTANT: keep config-like values as STRINGS so they don't become charts.
    eval_cfg = config.get("evaluation", {}) or {}
    gs_cv = eval_cfg.get("grid_search_cv_folds", None)
    fe_cv = eval_cfg.get("cv_folds", None)
    scoring = eval_cfg.get("grid_search_scoring", None)
    gs_rs = eval_cfg.get("grid_search_random_state", None)
    fe_rs = eval_cfg.get("final_eval_random_state", None)

    return {
        "meta/user": user,
        "meta/host": host,
        "meta/experiment": pipeline,  # e.g. grid_search_with_final_eval
        "meta/dataset": dataset_stem,
        "meta/dataset_path": dataset_path,
        "meta/classifier": clf_type,
        # optional: bump if you change experiment protocol
        "meta/protocol_version": "v1",
        #  config context as strings (NOT metrics)
        "meta/grid_search_cv_folds": str(gs_cv) if gs_cv is not None else "",
        "meta/final_eval_cv_folds": str(fe_cv) if fe_cv is not None else "",
        "meta/grid_search_scoring": str(scoring) if scoring is not None else "",
        "meta/grid_search_random_state": str(gs_rs) if gs_rs is not None else "",
        "meta/final_eval_random_state": str(fe_rs) if fe_rs is not None else "",
    }


def _log_artifacts(tracker, results_manager: ResultsManager, classifier_name: str):
    """
    Log artifact paths (CSV/PNG) that ResultsManager already writes.
    We log paths as strings so teammates can find outputs quickly.
    """
    if tracker is None:
        return

    out = Path(results_manager.output_dir)

    # NOTE: these filenames follow the ResultsManager.save_all_results(...) convention.
    # If the ResultsManager uses slightly different naming, adjust here.
    artifacts = {
        "artifacts/output_dir": str(out),
        "artifacts/classification_report_csv": str(out / f"{classifier_name}_classification_report.csv"),
        "artifacts/roc_curve_png": str(out / f"{classifier_name}_roc_curve.png"),
        "artifacts/confusion_matrix_png": str(out / f"{classifier_name}_confusion_matrix.png"),
        "artifacts/confusion_matrix_norm_true_png": str(out / f"{classifier_name}_confusion_matrix_norm_true.png"),
    }
    safe_log(tracker, artifacts)


def _log_only_numeric_metrics(tracker, payload: Dict[str, Any]):
    """
    Only log scalar numeric metrics to Trackio charts.

    This prevents config fields (like random seeds) or dicts (like best_params)
    from polluting plots and showing up as huge y-axis values.
    """
    if tracker is None:
        return

    clean: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            clean[k] = float(v)
            continue
        # allow numpy numeric scalars if present
        try:
            import numpy as np
            if isinstance(v, np.generic) and np.isscalar(v):
                clean[k] = float(v)
        except Exception:
            pass

    if clean:
        safe_log(tracker, clean)


def _log_metrics_block(tracker, *, pipeline: str, clf_type: str, metrics_obj: Any):
    """
    Log a clean, reviewer-friendly set of numeric metrics.

    - Only metrics/* keys are numeric => only these become charts.
    - Everything else is metadata or JSON strings.
    """
    if tracker is None:
        return

    # Core numeric metrics (these are what you want the dashboard to show)
    numeric = {
        # common
        "metrics/roc_auc": getattr(metrics_obj, "roc_auc", None),
        "metrics/final_roc_auc": getattr(metrics_obj, "roc_auc", None), # alias for clarity
        "metrics/cv_folds": getattr(metrics_obj, "cv_folds", None),
        "metrics/grid_search_best_score": getattr(metrics_obj, "best_score", None),
    }
    _log_only_numeric_metrics(tracker, numeric)

    # Structured non-numeric context (keep as strings so these DON'T plot)
    best_params = getattr(metrics_obj, "best_params", None)
    safe_log(tracker, {
        "meta/pipeline": pipeline,
        "meta/classifier_type": clf_type,
        "meta/classifier_name": str(getattr(metrics_obj, "classifier_name", "")),
        "meta/best_params_json": json.dumps(best_params, default=str) if best_params is not None else "",
    })


def _log_classification_summary_metrics(tracker, metrics_obj: Any):
    """
    Compute and log reviewer-friendly classification metrics:
      - accuracy
      - macro avg precision/recall/f1
      - weighted avg precision/recall/f1
      - per-class f1 + support

    Uses y_true/y_pred already stored in your metrics object.
    """
    if tracker is None:
        return

    y_true = getattr(metrics_obj, "y_true", None)
    y_pred = getattr(metrics_obj, "y_pred", None)
    if y_true is None or y_pred is None:
        return

    try:
        from sklearn.metrics import classification_report
    except Exception:
        return

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    payload: Dict[str, Any] = {}

    # overall accuracy
    if "accuracy" in report:
        payload["metrics/accuracy"] = report["accuracy"]

    # macro avg
    macro = report.get("macro avg", {}) or {}
    payload["metrics/precision_macro"] = macro.get("precision", None)
    payload["metrics/recall_macro"] = macro.get("recall", None)
    payload["metrics/f1_macro"] = macro.get("f1-score", None)

    # weighted avg
    wavg = report.get("weighted avg", {}) or {}
    payload["metrics/precision_weighted"] = wavg.get("precision", None)
    payload["metrics/recall_weighted"] = wavg.get("recall", None)
    payload["metrics/f1_weighted"] = wavg.get("f1-score", None)

    # per-class metrics (F1 + support)
    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(stats, dict):
            payload[f"metrics/f1_class/{label}"] = stats.get("f1-score", None)
            payload[f"metrics/support_class/{label}"] = stats.get("support", None)

    _log_only_numeric_metrics(tracker, payload)


def run_evaluation(config: dict, classifiers: list = None):
    """
    Run evaluation pipeline for specified classifiers (no grid search).

    Args:
        config: Configuration dictionary
        classifiers: List of classifier types to evaluate.
                    If None, uses classifier from config.
                    Options: ["logreg", "rf", "svm", "mlp"]
    """
    # Load and prepare data
    print("Loading dataset...")
    dataset_df = load_dataset_df(config)
    X, y = prepare_data(dataset_df)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Get unique class labels for reporting
    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    # Initialize ResultsManager
    results_manager = ResultsManager(

        config=config,
        class_names=class_names,
    )

    # Determine which classifiers to run
    if classifiers is None:
        classifiers = [config["model"]["classifier"]]

    # Cross-validation folds
    cv_folds = config.get("evaluation", {}).get("cv_folds", 5)

    # Run evaluation for each classifier
    for clf_type in classifiers:
        # one tracker run per classifier with intuitive run name + group
        run_name = _make_run_name(config, clf_type, pipeline="evaluation")
        group_name = _make_group_name(config, pipeline="evaluation")

        tracker = get_tracker(config, run_name=run_name, group=group_name)

        # NOTE: keeping “context” fields as STRINGS so they don't become charts.
        safe_log(tracker, {
            "meta/pipeline": "evaluation",
            "meta/classifier/type": str(clf_type),
            "meta/evaluation/cv_folds": str(cv_folds),
            "meta/data/n_samples": str(int(X.shape[0])),
            "meta/data/n_features": str(int(X.shape[1])),
            "meta/data/dataset_path": str(config.get("data", {}).get("dataset_path", "")),
        })

        # collaboration metadata
        safe_log(tracker, _run_metadata(config, clf_type=clf_type, pipeline="evaluation"))

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {clf_type}")
        print(f"{'=' * 60}")

        # Initialize classifier
        classifier = SKClassifier(clf_type, config)

        # Run evaluation
        metrics = classifier.evaluate_model(X, y, cv=cv_folds)

        # Log per-classifier metrics (clean dashboard block)
        _log_metrics_block(tracker, pipeline="evaluation", clf_type=clf_type, metrics_obj=metrics)

        # log precision/recall/f1/accuracy (macro + weighted + per-class)
        _log_classification_summary_metrics(tracker, metrics)

        # Convert to EvaluationResult for storage
        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds,
        )

        # Add to results manager
        results_manager.add_result(eval_result)

        # Save individual results
        results_manager.save_all_results(eval_result)

        # log artifact paths for collaboration
        _log_artifacts(tracker, results_manager, classifier_name=metrics.classifier_name)

        safe_log(tracker, {
            "meta/results/output_dir": str(results_manager.output_dir),
            "meta/run/status": "completed",
        })
        if tracker is not None:
            tracker.finish()

    # Save combined report if multiple classifiers
    if len(classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {results_manager.output_dir}")
    print(f"{'=' * 60}")

    return results_manager


def run_grid_search_experiment(
    config: dict,
    classifiers: Optional[List[str]] = None,
    custom_param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Run grid search with unbiased final evaluation for specified classifiers.

    This addresses the optimistic bias problem when you can't have a held-out test set:
    1. Grid search uses StratifiedKFold with random_state=42 to find best params
    2. Final evaluation uses StratifiedKFold with random_state=123 (fresh splits)

    Only the final unbiased results are saved.

    Args:
        config: Configuration dictionary
        classifiers: List of classifier types to evaluate.
                    If None, uses all classifiers with param_grids in config.
                    Options: ["logreg", "rf", "svm", "mlp"]
        custom_param_grids: Optional dict of custom param grids to override config.
                           Format: {"logreg": {"C": [0.1, 1], ...}, ...}

    Returns:
        ResultsManager with all results
    """
    # Load and prepare data
    print("Loading dataset...")
    dataset_df = load_dataset_df(config)
    X, y = prepare_data(dataset_df)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Get unique class labels for reporting
    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    # Initialize ResultsManager (single output dir for whole experiment session)
    results_manager = ResultsManager(
        config=config,
        class_names=class_names,
    )

    # Get param grids from config
    config_param_grids = config.get("model", {}).get("param_grids", {})

    # Merge with custom param grids (custom takes precedence)
    if custom_param_grids:
        param_grids = {**config_param_grids, **custom_param_grids}
    else:
        param_grids = config_param_grids

    # Determine which classifiers to run
    if classifiers is None:
        classifiers = list(param_grids.keys())

    # Filter to classifiers that have param grids
    valid_classifiers = []
    for clf_type in classifiers:
        if clf_type in param_grids:
            valid_classifiers.append(clf_type)
        else:
            print(f"Warning: No param_grid found for {clf_type}, skipping.")

    if not valid_classifiers:
        raise ValueError(
            "No classifiers with param_grids to evaluate. "
            "Add param_grids to config or provide custom_param_grids."
        )

    # Get evaluation settings
    eval_config = config.get("evaluation", {})
    grid_search_cv = eval_config.get("grid_search_cv_folds", 5)
    final_eval_cv = eval_config.get("cv_folds", 5)
    scoring = eval_config.get("grid_search_scoring", "roc_auc")
    grid_search_random_state = eval_config.get("grid_search_random_state", 42)
    final_eval_random_state = eval_config.get("final_eval_random_state", 123)

    # Store best params for summary
    best_params_summary = {}

    # Run grid search + final eval for each classifier
    for clf_type in valid_classifiers:
        # (OPTION B) one tracker run per classifier with intuitive run name + group
        run_name = _make_run_name(config, clf_type, pipeline="grid_search_with_final_eval")
        group_name = _make_group_name(config, pipeline="grid_search_with_final_eval")

        tracker = get_tracker(config, run_name=run_name, group=group_name)

        # NOTE: keeping “context” fields as STRINGS so they don't become charts.
        safe_log(tracker, {
            "meta/pipeline": "grid_search_with_final_eval",
            "meta/classifier/type": str(clf_type),
            "meta/evaluation/grid_search_cv_folds": str(grid_search_cv),
            "meta/evaluation/final_eval_cv_folds": str(final_eval_cv),
            "meta/evaluation/scoring": str(scoring),
            "meta/evaluation/grid_search_random_state": str(grid_search_random_state),
            "meta/evaluation/final_eval_random_state": str(final_eval_random_state),
            "meta/data/n_samples": str(int(X.shape[0])),
            "meta/data/n_features": str(int(X.shape[1])),
            "meta/data/dataset_path": str(config.get("data", {}).get("dataset_path", "")),
        })

        # collaboration metadata
        safe_log(tracker, _run_metadata(config, clf_type=clf_type, pipeline="grid_search_with_final_eval"))

        param_grid = param_grids[clf_type]

        # Initialize classifier
        classifier = SKClassifier(clf_type, config)

        # Run grid search with unbiased final evaluation
        metrics = classifier.grid_search_with_final_eval(
            X,
            y,
            param_grid=param_grid,
            grid_search_cv=grid_search_cv,
            final_eval_cv=final_eval_cv,
            scoring=scoring,
            grid_search_random_state=grid_search_random_state,
            final_eval_random_state=final_eval_random_state,
            verbose=True,
        )

        best_params_summary[clf_type] = metrics.best_params

        # Log numeric metrics (clean dashboard block)
        _log_metrics_block(tracker, pipeline="grid_search_with_final_eval", clf_type=clf_type, metrics_obj=metrics)

        #  log precision/recall/f1/accuracy (macro + weighted + per-class)
        _log_classification_summary_metrics(tracker, metrics)

        # Convert to EvaluationResult for storage
        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds,
            additional_metrics={"best_params": metrics.best_params},
        )

        # Add to results manager
        results_manager.add_result(eval_result)

        # Save individual results
        results_manager.save_all_results(eval_result)

        # log artifact paths for collaboration
        _log_artifacts(tracker, results_manager, classifier_name=metrics.classifier_name)

        safe_log(tracker, {
            "meta/results/output_dir": str(results_manager.output_dir),
            "meta/run/status": "completed",
        })
        if tracker is not None:
            tracker.finish()

    # Save combined report if multiple classifiers
    if len(valid_classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    # Save best params summary (once per session)
    _save_best_params_summary(results_manager.output_dir, best_params_summary)

    print(f"\n{'=' * 60}")
    print("Grid Search Experiment Complete!")
    print(f"All results saved to: {results_manager.output_dir}")
    print(f"{'=' * 60}")

    # Print best params summary
    print("\nBest Parameters Summary:")
    for clf_type, params in best_params_summary.items():
        print(f"  {clf_type}: {params}")

    return results_manager


def _save_best_params_summary(output_dir: Path, best_params: Dict[str, Dict]):
    """Save a summary of best parameters to a JSON file."""
    summary_path = output_dir / "best_params_summary.json"
    with open(summary_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    print(f"\n✓ Best params summary saved: {summary_path}")


if __name__ == "__main__":
    # Load config
    config = load_config()

    # ===== OPTION 1: Simple evaluation (no hyperparameter tuning) =====
    #run_evaluation(config)

    # ===== OPTION 2: Simple evaluation with multiple classifiers =====
    #run_evaluation(config, classifiers=["logreg", "rf", "svm", "mlp"])

    # ===== OPTION 3: Grid search with unbiased final evaluation =====
    # This is the recommended approach when you can't have a held-out test set.
    # It finds best hyperparameters, then evaluates on fresh CV splits.
    run_grid_search_experiment(config, classifiers=["logreg", "rf", "svm", "mlp"])

    # ===== OPTION 4: Grid search for specific classifiers =====
    #run_grid_search_experiment(config, classifiers=["logreg", "mlp"])

    # ===== OPTION 5: Grid search with custom param grids =====
    # custom_grids = {
    #     "logreg": {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]},
    #     "rf": {"n_estimators": [100, 200], "max_depth": [10, 20]}
    # }
    # run_grid_search_experiment(config, custom_param_grids=custom_grids)
