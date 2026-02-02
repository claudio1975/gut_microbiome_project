# Classifier module integrating foundation model and classification head
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from typing import Dict, Any
from utils.evaluation_utils import EvaluationMetrics
from omegaconf import DictConfig
from rich.console import Console


class SKClassifier:
    """Scikit-learn based classifier with cross-validation and grid search."""

    # Mapping of classifier types to their names
    CLASSIFIER_NAMES = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "svm": "SVM",
        "mlp": "MLP",
    }

    def __init__(
        self, classifier_type: str, config: DictConfig, console: Console = None
    ):
        self.classifier_type = classifier_type
        self.config = config
        self.console = console or Console()
        self.use_scaler = config.model.use_scaler
        self.pipeline = self._init_pipeline()
        self.best_params = None

    @property
    def name(self) -> str:
        """Get human-readable classifier name."""
        base_name = self.CLASSIFIER_NAMES.get(
            self.classifier_type, self.classifier_type
        )
        return f"{base_name} (scaled)" if self.use_scaler else base_name

    @property
    def classifier(self):
        """Get the pipeline (for compatibility)."""
        return self.pipeline

    @classifier.setter
    def classifier(self, value):
        """Set the pipeline (for compatibility)."""
        self.pipeline = value

    def _init_base_classifier(self, params: Dict[str, Any] = None):
        """Initialize the base classifier with optional parameters."""
        params = params or {}

        if self.classifier_type == "logreg":
            return LogisticRegression(**params)
        elif self.classifier_type == "rf":
            return RandomForestClassifier(**params)
        elif self.classifier_type == "svm":
            # Use probability=True for ROC curve support
            return SVC(probability=True, **params)
        elif self.classifier_type == "mlp":
            return MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def _init_pipeline(self, classifier_params: Dict[str, Any] = None):
        """Initialize pipeline with optional StandardScaler."""
        base_clf = self._init_base_classifier(classifier_params)

        if self.use_scaler:
            return Pipeline([("scaler", StandardScaler()), ("classifier", base_clf)])
        else:
            # Still use pipeline for consistency, just without scaler
            return Pipeline([("classifier", base_clf)])

    def _prefix_params_for_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefix parameter names for sklearn Pipeline (classifier__param)."""
        return {f"classifier__{k}": v for k, v in params.items()}

    def _unprefix_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove pipeline prefix from parameter names."""
        return {k.replace("classifier__", ""): v for k, v in params.items()}

    def set_params(self, **params):
        """Set classifier parameters and reinitialize pipeline."""
        self.pipeline = self._init_pipeline(params)
        self.best_params = params

    def evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int,
        random_state: int = 42,
        verbose: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate model using Stratified K-Fold Cross-Validation.

        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
            verbose: Whether to self.console.print results

        Returns:
            EvaluationMetrics object containing all evaluation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # Get predictions using pipeline
        y_pred = cross_val_predict(self.pipeline, X, y, cv=skf, method="predict")

        # Get probability predictions (for ROC-AUC)
        y_prob = None
        try:
            y_prob_all = cross_val_predict(
                self.pipeline, X, y, cv=skf, method="predict_proba"
            )
            # For binary classification, take probability of positive class
            if y_prob_all.shape[1] == 2:
                y_prob = y_prob_all[:, 1]
            else:
                y_prob = y_prob_all
        except Exception as e:
            if verbose:
                self.console.print(
                    f"Warning: Could not get probability predictions: {e}"
                )

        # Create metrics container
        metrics = EvaluationMetrics(
            classifier_name=self.name,
            y_true=y,
            y_pred=y_pred,
            y_prob=y_prob,
            cv_folds=cv,
            best_params=self.best_params,
        )

        if verbose:
            self.console.print(f"\n{'=' * 60}")
            self.console.print(f"Evaluation Results: {self.name}", style="bold green")
            if self.use_scaler:
                self.console.print("Preprocessing: StandardScaler", style="dim")
            if self.best_params:
                self.console.print(f"Parameters: {self.best_params}", style="dim")
            self.console.print(f"{'=' * 60}")
            self.console.print(
                f"\nStratified {cv}-Fold Cross-Validation Classification Report:",
                style="bold",
            )
            self.console.print(classification_report(y, y_pred), style="cyan")

            if metrics.roc_auc is not None:
                self.console.print(
                    f"ROC-AUC Score: {metrics.roc_auc:.4f}", style="bold magenta"
                )

        return metrics

    def grid_search_with_final_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, Any],
        grid_search_cv: int = 5,
        final_eval_cv: int = 5,
        scoring: str = "roc_auc",
        grid_search_random_state: int = 42,
        final_eval_random_state: int = 123,
        verbose: bool = True,
    ) -> EvaluationMetrics:
        """
        Run grid search for hyperparameter tuning, then perform unbiased final evaluation.

        This method addresses optimistic bias by:
        1. Using one random state for grid search CV (inner loop)
        2. Using a DIFFERENT random state for final evaluation CV

        This ensures the final performance estimate is not inflated by
        using the same CV splits that selected the best hyperparameters.

        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for grid search
            grid_search_cv: Number of CV folds for grid search
            final_eval_cv: Number of CV folds for final evaluation
            scoring: Scoring metric for grid search
            grid_search_random_state: Random state for grid search CV
            final_eval_random_state: DIFFERENT random state for final evaluation
            verbose: Whether to self.console.print results

        Returns:
            EvaluationMetrics from the final unbiased evaluation
        """
        if verbose:
            self.console.print(f"\n{'=' * 60}")
            self.console.print(
                f"Grid Search + Final Evaluation: {self.name}", style="bold green"
            )
            if self.use_scaler:
                self.console.print("Using StandardScaler in pipeline", style="dim")
            self.console.print(f"{'=' * 60}")
            self.console.print(
                f"\nPhase 1: Grid Search (random_state={grid_search_random_state})",
                style="bold",
            )
            self.console.print(f"Parameter grid: {param_grid}", style="dim")

        # Prefix params for Pipeline (classifier__param_name)
        pipeline_param_grid = self._prefix_params_for_pipeline(param_grid)

        # Phase 1: Grid Search with one random state
        skf_grid = StratifiedKFold(
            n_splits=grid_search_cv, shuffle=True, random_state=grid_search_random_state
        )

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=pipeline_param_grid,
            scoring=scoring,
            cv=skf_grid,
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )

        grid_search.fit(X, y)

        # Get best params and remove pipeline prefix for clean storage
        best_params_prefixed = grid_search.best_params_
        best_params = self._unprefix_params(best_params_prefixed)
        best_score = grid_search.best_score_

        if verbose:
            self.console.print(f"\nBest parameters: {best_params}", style="bold yellow")
            self.console.print(
                f"Best {scoring} score (grid search CV): {best_score:.4f}",
                style="bold yellow",
            )

        # Phase 2: Final Unbiased Evaluation with DIFFERENT random state
        if verbose:
            self.console.print(
                f"\nPhase 2: Final Unbiased Evaluation (random_state={final_eval_random_state})",
                style="bold",
            )
            self.console.print(
                "Re-instantiating classifier with best params and fresh CV splits...",
                style="dim",
            )

        # Create fresh pipeline with best params
        self.set_params(**best_params)

        # Evaluate with DIFFERENT random state to remove optimistic bias
        metrics = self.evaluate_model(
            X,
            y,
            cv=final_eval_cv,
            random_state=final_eval_random_state,
            verbose=verbose,
        )

        if verbose:
            self.console.print(f"\n{'=' * 60}")
            self.console.print(
                f"Grid Search CV {scoring}: {best_score:.4f}", style="bold yellow"
            )
            self.console.print(
                f"Final Unbiased {scoring}: {metrics.roc_auc:.4f}"
                if metrics.roc_auc
                else ""
            )
            self.console.print(f"{'=' * 60}")

        return metrics

    def train_test_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_grid: Dict[str, Any] = None,
        grid_search_cv: int = 5,
        scoring: str = "roc_auc",
        grid_search_random_state: int = 42,
        verbose: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate a classifier using a held-out test set.

        Optionally runs GridSearchCV on the training set first to find
        the best hyperparameters, then fits on the full training set and
        predicts on the test set.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_test: Test feature matrix
            y_test: Test labels
            param_grid: Optional parameter grid for grid search (on train set)
            grid_search_cv: Number of CV folds for grid search
            scoring: Scoring metric for grid search
            grid_search_random_state: Random state for grid search CV
            verbose: Whether to print progress

        Returns:
            EvaluationMetrics with cv_folds=0 (signals holdout evaluation)
        """
        if verbose:
            self.console.print(f"\n{'=' * 60}")
            self.console.print(
                f"Train/Test Evaluation: {self.name}", style="bold green"
            )
            self.console.print(
                f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}",
                style="dim",
            )
            self.console.print(f"{'=' * 60}")

        # Phase 1: Optional grid search on training set
        if param_grid is not None:
            if verbose:
                self.console.print(
                    f"\nPhase 1: Grid Search on train set "
                    f"(random_state={grid_search_random_state})",
                    style="bold",
                )
                self.console.print(f"Parameter grid: {param_grid}", style="dim")

            pipeline_param_grid = self._prefix_params_for_pipeline(param_grid)

            skf_grid = StratifiedKFold(
                n_splits=grid_search_cv,
                shuffle=True,
                random_state=grid_search_random_state,
            )

            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=pipeline_param_grid,
                scoring=scoring,
                cv=skf_grid,
                n_jobs=-1,
                verbose=1 if verbose else 0,
            )

            grid_search.fit(X_train, y_train)

            best_params = self._unprefix_params(grid_search.best_params_)
            best_score = grid_search.best_score_

            if verbose:
                self.console.print(
                    f"\nBest parameters: {best_params}", style="bold yellow"
                )
                self.console.print(
                    f"Best {scoring} score (grid search CV): {best_score:.4f}",
                    style="bold yellow",
                )

            self.set_params(**best_params)

        # Phase 2: Fit on full training set and predict on test set
        if verbose:
            self.console.print(
                "\nPhase 2: Fit on train set, evaluate on held-out test set",
                style="bold",
            )

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        y_prob = None
        try:
            y_prob_all = self.pipeline.predict_proba(X_test)
            if y_prob_all.shape[1] == 2:
                y_prob = y_prob_all[:, 1]
            else:
                y_prob = y_prob_all
        except Exception as e:
            if verbose:
                self.console.print(
                    f"Warning: Could not get probability predictions: {e}"
                )

        metrics = EvaluationMetrics(
            classifier_name=self.name,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            cv_folds=0,
            best_params=self.best_params,
        )

        if verbose:
            self.console.print(
                f"\nHoldout Test Set Classification Report:", style="bold"
            )
            self.console.print(
                classification_report(y_test, y_pred), style="cyan"
            )
            if metrics.roc_auc is not None:
                self.console.print(
                    f"ROC-AUC Score: {metrics.roc_auc:.4f}", style="bold magenta"
                )

        return metrics

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the pipeline on training data."""
        self.console.print(f"Fitting {self.name}...", style="bold green")
        self.pipeline.fit(X, y)
        self.console.print("Pipeline fitted.", style="success")

    def save_model(self, path: str) -> None:
        """Save the trained pipeline to disk."""

        self.console.print("Saving model...", style="info")
        joblib.dump(self.pipeline, path)
        self.console.print(f"Model saved to {path}", style="path")

    def load_model(self, path: str) -> None:
        """Load a trained pipeline from disk."""

        self.console.print(f"Loading model from {path}...", style="info")
        self.pipeline = joblib.load(path)
        self.console.print("Model loaded.", style="path")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions on new data."""
        return self.pipeline.predict_proba(X)
