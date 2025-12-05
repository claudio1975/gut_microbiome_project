# Classifier module integrating foundation model and classification head
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

class SKClassifier:
    def __init__(self, classifier_type: str, config: Dict[str, Any]):
        self.classifier_type = classifier_type
        self.config = config['model']
        self.classifier = self.init_classifier()
        
    def init_classifier(self):
        if self.classifier_type == "logreg":
            return LogisticRegression(class_weight=self.config['logistic_regression_params']['class_weight'], 
                                      random_state=self.config['logistic_regression_params']['random_state'],
                                      solver=self.config['logistic_regression_params']['solver'],
                                      max_iter=self.config['logistic_regression_params']['max_iter'])
        elif self.classifier_type == "rf":
            return RandomForestClassifier()
        elif self.classifier_type == "svm":
            return SVC()
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5):
        y_pred = cross_val_predict(
            self.classifier,
            X,
            y,
            cv=k,
            method="predict"
        )
        # Probability predictions (for ROC-AUC)
        y_prob = cross_val_predict(
            self.classifier,
            X,
            y,
            cv=k,
            method="predict_proba"
        )[:, 1]   # take probability of positive class

        # -----------------------------
        # METRICS
        # -----------------------------

        print("K-Fold Classification Report:")
        print(classification_report(y, y_pred))

        roc_auc = roc_auc_score(y, y_prob)
        print("K-Fold ROC-AUC Score:", roc_auc)

        # plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        # save results to file
        with open("results.txt", "w") as f:
            f.write(classification_report(y, y_pred))
            f.write("\n")
            f.write("ROC-AUC Score: " + str(roc_auc))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classifier.fit(X, y)
        
    def predict(self, X: np.ndarray):
        return self.classifier.predict(X)