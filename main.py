# main.py
# Entry point for model training/evaluation + Trackio logging.

from __future__ import annotations

import json
import hydra
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from omegaconf import DictConfig, OmegaConf
from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils.data_utils import prepare_data
from utils.evaluation_utils import EvaluationResult, ResultsManager
from utils.generic_utils import RichConsoleManager
from utils.tracking_utils import (
    get_tracker,
    safe_log,
    _apply_tracking_routing,
    _confirm_publish_to_main,
    _dataset_family_and_id,
    _get_tags,
    _log_artifacts,
    _log_classification_summary_metrics,
    _log_eval_images,
    _log_metrics_block,
    _make_group_name,
    _make_run_name,
    _patch_httpx_nan_guard,
    _run_metadata,
    _save_best_params_summary,
    _log_session_outputs_summary_run,
)


def run_evaluation(
    config: dict,
    console: Console = Console,
    classifiers: Optional[List[str]] = None,
):
    console.print("Loading dataset...", style="bold")
    dataset_df = load_dataset_df(config, console=console)
    X, y = prepare_data(dataset_df)
    console.print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    results_manager = ResultsManager(
        config=config, console=console, class_names=class_names
    )

    if classifiers is None:
        classifiers = config.model.classifier

    cv_folds = config.evaluation.cv_folds
    dataset_family, data_id = _dataset_family_and_id(config.data.dataset_path)

    for clf_type in classifiers:
        run_name = _make_run_name(config, clf_type, pipeline="evaluation")
        group_name = _make_group_name(config, pipeline="evaluation")
        tracker = get_tracker(
            config, run_name=run_name, group=group_name, console=console
        )

        tags = _get_tags(
            config,
            dataset_family=dataset_family,
            data_id=data_id,
            pipeline="evaluation",
            clf_type=clf_type,
        )
        safe_log(tracker, {"meta/tags_json": json.dumps(tags)})
        for t in tags:
            safe_log(tracker, {f"meta/tag/{t}": "1"})

        safe_log(
            tracker,
            {
                "meta/pipeline": "evaluation",
                "meta/classifier/type": str(clf_type),
                "meta/evaluation/cv_folds": str(cv_folds),
                "meta/data/n_samples": str(int(X.shape[0])),
                "meta/data/n_features": str(int(X.shape[1])),
                "meta/data/dataset_path": config.data.dataset_path,
            },
        )
        safe_log(
            tracker, _run_metadata(config, clf_type=clf_type, pipeline="evaluation")
        )

        classifier = SKClassifier(clf_type, config, console=console)
        metrics = classifier.evaluate_model(X, y, cv=cv_folds)

        _log_metrics_block(
            tracker, pipeline="evaluation", clf_type=clf_type, metrics_obj=metrics
        )
        _log_classification_summary_metrics(tracker, metrics)

        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds,
        )

        results_manager.add_result(eval_result)
        results_manager.save_all_results(eval_result)

        _log_eval_images(
            tracker, Path(results_manager.output_dir), metrics.classifier_name
        )
        _log_artifacts(
            tracker, results_manager, classifier_name=metrics.classifier_name
        )

        safe_log(
            tracker,
            {
                "meta/results/output_dir": str(results_manager.output_dir),
                "meta/run/status": "completed",
            },
        )
        if tracker is not None:
            tracker.finish()

    if len(classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()
        _log_session_outputs_summary_run(config, results_manager)

    console.print(f"\nAll results saved to: {results_manager.output_dir}")
    return results_manager


def run_grid_search_experiment(
    config: dict,
    console: Console = Console,
    classifiers: Optional[List[str]] = None,
    custom_param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
):
    console.print("Loading dataset...", style="bold")
    dataset_df = load_dataset_df(config, console=console)
    X, y = prepare_data(dataset_df)
    console.print(
        f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features", style="success"
    )

    # Get class names
    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    # Initialize ResultsManager
    results_manager = ResultsManager(
        config=config, console=console, class_names=class_names
    )

    # Determine classifiers and param grids
    config_param_grids = config.model.param_grids
    param_grids = {**config_param_grids, **(custom_param_grids or {})}

    if classifiers is None:
        classifiers = list(param_grids.keys())

    valid_classifiers = [c for c in classifiers if c in param_grids]
    if not valid_classifiers:
        raise ValueError("No classifiers with param_grids to evaluate.")

    # Extract evaluation settings
    grid_search_cv = config.evaluation.grid_search_cv_folds
    final_eval_cv = config.evaluation.cv_folds
    scoring = config.evaluation.grid_search_scoring
    grid_search_random_state = config.evaluation.grid_search_random_state
    final_eval_random_state = config.evaluation.final_eval_random_state

    # Extract dataset info for tagging
    dataset_family, data_id = _dataset_family_and_id(config.data.dataset_path)
    best_params_summary: Dict[str, Dict[str, Any]] = {}

    # Run experiments for each classifier
    for clf_type in valid_classifiers:
        run_name = _make_run_name(
            config, clf_type, pipeline="grid_search_with_final_eval"
        )
        group_name = _make_group_name(config, pipeline="grid_search_with_final_eval")
        tracker = get_tracker(config, run_name=run_name, group=group_name)

        # Log tags
        tags = _get_tags(
            config,
            dataset_family=dataset_family,
            data_id=data_id,
            pipeline="grid_search_with_final_eval",
            clf_type=clf_type,
        )
        safe_log(tracker, {"meta/tags_json": json.dumps(tags)})
        for t in tags:
            safe_log(tracker, {f"meta/tag/{t}": "1"})

        safe_log(
            tracker,
            {
                "meta/pipeline": "grid_search_with_final_eval",
                "meta/classifier/type": str(clf_type),
                "meta/evaluation/grid_search_cv_folds": str(grid_search_cv),
                "meta/evaluation/final_eval_cv_folds": str(final_eval_cv),
                "meta/evaluation/scoring": str(scoring),
                "meta/evaluation/grid_search_random_state": str(
                    grid_search_random_state
                ),
                "meta/evaluation/final_eval_random_state": str(final_eval_random_state),
                "meta/data/n_samples": str(int(X.shape[0])),
                "meta/data/n_features": str(int(X.shape[1])),
                "meta/data/dataset_path": str(
                    config.get("data", {}).get("dataset_path", "")
                ),
            },
        )
        safe_log(
            tracker,
            _run_metadata(
                config, clf_type=clf_type, pipeline="grid_search_with_final_eval"
            ),
        )

        # Run grid search with final evaluation
        classifier = SKClassifier(clf_type, config)
        metrics = classifier.grid_search_with_final_eval(
            X,
            y,
            param_grid=param_grids[clf_type],
            grid_search_cv=grid_search_cv,
            final_eval_cv=final_eval_cv,
            scoring=scoring,
            grid_search_random_state=grid_search_random_state,
            final_eval_random_state=final_eval_random_state,
            verbose=True,
        )

        best_params_summary[clf_type] = metrics.best_params

        # Log metrics and results
        _log_metrics_block(
            tracker,
            pipeline="grid_search_with_final_eval",
            clf_type=clf_type,
            metrics_obj=metrics,
        )
        _log_classification_summary_metrics(tracker, metrics)

        # Save evaluation result
        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds,
            additional_metrics={"best_params": metrics.best_params},
        )

        results_manager.add_result(eval_result)
        results_manager.save_all_results(eval_result)

        # Log images and artifacts
        _log_eval_images(
            tracker, Path(results_manager.output_dir), metrics.classifier_name
        )
        _log_artifacts(
            tracker, results_manager, classifier_name=metrics.classifier_name
        )

        safe_log(
            tracker,
            {
                "meta/results/output_dir": str(results_manager.output_dir),
                "meta/run/status": "completed",
            },
        )
        if tracker is not None:
            tracker.finish()

    if len(valid_classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    _save_best_params_summary(Path(results_manager.output_dir), best_params_summary)
    _log_session_outputs_summary_run(config, results_manager)

    console.print(
        f"\nAll results saved to: {results_manager.output_dir}", style="success"
    )
    return results_manager


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    # Apply HTTPX NaN guard patch
    _patch_httpx_nan_guard()

    _, data_id = _dataset_family_and_id(cfg.data.dataset_path)
    if cfg.tracking.publish_to_main:
        # Confirm before proceeding
        _confirm_publish_to_main()
    if cfg.tracking.enabled:
        # Apply tracking routing based on publish_to_main flag
        _apply_tracking_routing(
            cfg,
            data_id=data_id,
            console=console,
        )
        console.print(f"Tracking configured. Project: {cfg.tracking.project}")
    if cfg.mode.eval_only:
        # Simple evaluation (no hyperparameter tuning)
        console.print("Running evaluation only...", style="warning")
        run_evaluation(config=cfg, console=console, classifiers=cfg.model.classifier)
    else:
        # Default behavior: grid search + final eval
        console.print("Running grid search experiment...", style="warning")
        run_grid_search_experiment(
            config=cfg,
            console=console,
            classifiers=cfg.model.classifier,
        )


if __name__ == "__main__":
    # Execute the main entry point function
    main()
