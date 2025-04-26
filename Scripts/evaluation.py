import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

class ModelEvaluator:
    def __init__(self, models, X_test, y_test, feature_names, vis_dir, model_dir):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.vis_dir = vis_dir
        self.model_dir = model_dir
        self.class_names = np.unique(y_test)
        os.makedirs(vis_dir, exist_ok=True)

    def evaluate_all(self):
        metrics_results = []
        all_predictions = {}

        for name, model in self.models:
            print(f"\nEvaluating {name}...")
            y_pred, y_proba = self._get_predictions(name, model)
            if y_pred is None: 
                continue
                
            all_predictions[name] = y_pred

            metrics = self._calculate_metrics(y_pred, y_proba, name)
            metrics_results.append(metrics)

            self._generate_classification_report(y_pred, name)
            self._generate_confusion_matrix(y_pred, name)
            
            if y_proba is not None:
                self._generate_roc_curves(y_proba, name)

        self._save_and_compare_metrics(metrics_results)
        self._generate_feature_importances()

        return pd.DataFrame(metrics_results), all_predictions

    def _get_predictions(self, name, model):
        if "XGBoost" in name:
            try:
                encoder = joblib.load(os.path.join(self.model_dir, 'XGBoost_label_encoder.pkl'))
                y_pred = encoder.inverse_transform(model.predict(self.X_test))
                y_proba = model.predict_proba(self.X_test)
                return y_pred, y_proba
            except Exception as e:
                print(f"Error with XGBoost evaluation: {e}")
                return None, None
        else:
            y_pred = model.predict(self.X_test)
            try:
                y_proba = model.predict_proba(self.X_test)
            except:
                y_proba = None
            return y_pred, y_proba

    def _calculate_metrics(self, y_pred, y_proba, name):
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1': f1_score(self.y_test, y_pred, average='weighted'),
            'ROC AUC': None,
            'Top Class Accuracy': None
        }

        if y_proba is not None:
            try:
                y_test_bin = label_binarize(self.y_test, classes=self.class_names)
                metrics['ROC AUC'] = roc_auc_score(
                    y_test_bin, y_proba, multi_class='ovr', average='weighted'
                )
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")

        top_class = pd.Series(self.y_test).value_counts().index[0]
        top_mask = self.y_test == top_class
        metrics['Top Class Accuracy'] = accuracy_score(
            self.y_test[top_mask], y_pred[top_mask])

        return metrics

    def _generate_confusion_matrix(self, y_pred, name):
        top_classes = pd.Series(self.y_test).value_counts().nlargest(10).index
        y_true_grouped = np.where(np.isin(self.y_test, top_classes), self.y_test, 'Other')
        y_pred_grouped = np.where(np.isin(y_pred, top_classes), y_pred, 'Other')

        cm = confusion_matrix(
            y_true_grouped, y_pred_grouped,
            labels=list(top_classes)+['Other'],
            normalize='true'
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=list(top_classes)+['Other'],
                    yticklabels=list(top_classes)+['Other'])
        plt.title(f'Normalized Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()

    def _generate_roc_curves(self, y_proba, name):
        top_classes = pd.Series(self.y_test).value_counts().nlargest(5).index
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
        
        for i, cls in enumerate(top_classes):
            y_true_bin = (self.y_test == cls).astype(int)
            cls_idx = np.where(self.class_names == cls)[0][0]
            fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=next(colors), lw=2,
                    label=f'Class {cls} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.vis_dir, f'roc_curve_{name.replace(" ", "_")}.png'))
        plt.close()

    def _save_and_compare_metrics(self, metrics_results):
        metrics_df = pd.DataFrame(metrics_results)
        metrics_df.to_csv(os.path.join(self.vis_dir, 'model_metrics.csv'), index=False)

        plt.figure(figsize=(14, 8))
        metrics_df.set_index('Model').plot(kind='bar', rot=45)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'model_comparison.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df.set_index('Model'), annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title('Model Metrics Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'metrics_heatmap.png'))
        plt.close()

    def _generate_feature_importances(self):
        for name, model in self.models:
            if hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:20] 
                    
                    plt.figure(figsize=(12, 8))
                    plt.title(f"Feature Importances - {name}")
                    plt.bar(range(len(indices)), importances[indices], align="center")
                    plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=90)
                    plt.xlim([-1, len(indices)])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.vis_dir, f'feature_importance_{name.replace(" ", "_")}.png'))
                    plt.close()
                except Exception as e:
                    print(f"Could not generate feature importance for {name}: {e}")


def evaluate_models(models, X_test, y_test, feature_names, vis_dir, model_dir):
    evaluator = ModelEvaluator(models, X_test, y_test, feature_names, vis_dir, model_dir)
    return evaluator.evaluate_all()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,
        n_classes=5,
        n_features=20,
        n_informative=8,   
        n_redundant=5,    
        flip_y=0.15,       
        class_sep=0.8,       
        random_state=42
    )
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    xgb_encoder = LabelEncoder()
    y_train_encoded = xgb_encoder.fit_transform(y_train)
    os.makedirs("Models", exist_ok=True)
    joblib.dump(xgb_encoder, 'Models/XGBoost_label_encoder.pkl')
    
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000,
            random_state=42
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )),
        ("XGBoost", xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42
        ))
    ]
    
    for name, model in models:
        print(f"\nTraining {name}...")
        if "XGBoost" in name:
            model.fit(X_train, y_train_encoded)
        else:
            model.fit(X_train, y_train)
    
    metrics, predictions = evaluate_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        vis_dir="Visualizations/Evaluation",
        model_dir="Models"
    )
    
    print("\nEvaluation complete!")
    print("Metrics:")
    print(metrics)