# Main script to run train + evaluation

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils.evaluation_utils import ResultsManager, EvaluationResult
from utils.data_utils import load_config, prepare_data


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
        results_output_dir=config["evaluation"]["results_output_dir"],
        dataset_path=config["data"]["dataset_path"],
        class_names=class_names,
    )

    # Determine which classifiers to run
    if classifiers is None:
        classifiers = [config["model"]["classifier"]]

    # Cross-validation folds
    cv_folds = config.get("evaluation", {}).get("cv_folds", 5)

    # Run evaluation for each classifier
    for clf_type in classifiers:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {clf_type}")
        print(f"{'=' * 60}")

        # Initialize classifier
        classifier = SKClassifier(clf_type, config)

        # Run evaluation
        metrics = classifier.evaluate_model(X, y, cv=cv_folds)

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

    # Initialize ResultsManager
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

    # Save combined report if multiple classifiers
    if len(valid_classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    # Save best params summary
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
    # run_evaluation(config)

    # ===== OPTION 2: Simple evaluation with multiple classifiers =====
    # run_evaluation(config, classifiers=["logreg", "rf", "svm", "mlp"])

    # ===== OPTION 3: Grid search with unbiased final evaluation =====
    # This is the recommended approach when you can't have a held-out test set.
    # It finds best hyperparameters, then evaluates on fresh CV splits.
    run_grid_search_experiment(config, classifiers=["logreg", "rf", "svm", "mlp"])

    # ===== OPTION 4: Grid search for specific classifiers =====
    # run_grid_search_experiment(config, classifiers=["logreg", "mlp"])

    # ===== OPTION 5: Grid search with custom param grids =====
    # custom_grids = {
    #     "logreg": {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]},
    #     "rf": {"n_estimators": [100, 200], "max_depth": [10, 20]}
    # }
    # run_grid_search_experiment(config, custom_param_grids=custom_grids)
