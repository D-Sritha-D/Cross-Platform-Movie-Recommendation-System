import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve, precision_score, 
                            recall_score, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
import xgboost as xgb
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Train, tune, evaluate and visualize recommendation models')
    parser.add_argument('--data', type=str, default='Data/selected_features_for_recommendation.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for models and visualizations')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run evaluation on existing models')
    args = parser.parse_args()
    
    model_dir = os.path.join(args.output, 'Models')
    vis_dir = os.path.join(args.output, 'Visualizations/Model_Analysis')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nModel Training, Tuning, and Evaluation Pipeline")
    print(f"==============================================")
    print(f"Data file: {args.data}")
    print(f"Hyperparameter tuning: {'Yes' if args.tune else 'No'}")
    print(f"Output directory: {args.output}")
    print(f"Skip training: {'Yes' if args.skip_training else 'No'}")
    print(f"==============================================\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    X, y, feature_names = preprocess_data(args.data)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        print("Successfully split data with stratification")
    except ValueError as e:
        print(f"Warning: Could not stratify: {e}")
        print("Falling back to non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    models = []
    
    if not args.skip_training:
        # Step 2: Train models
        print("\nStep 2: Training models...")
        models = train_models(X_train, y_train, X_test, y_test, model_dir)
        
        # Step 3: Hyperparameter tuning (optional)
        if args.tune:
            print("\nStep 3: Performing hyperparameter tuning...")
            tuned_models = tune_hyperparameters(X_train, y_train, X_test, y_test, model_dir)
            models.extend(tuned_models)
    else:
        print("\nSkipping training, loading existing models...")
        models = load_models(model_dir)
        if not models:
            print("No models found. Please run without --skip_training first.")
            return
    
    # Step 4: Basic model evaluation
    print("\nStep 4: Evaluating models and creating basic visualizations...")
    evaluate_models(models, X_test, y_test, feature_names, vis_dir, model_dir)
    
    # Step 5: Advanced visualizations
    print("\nStep 5: Creating advanced visualizations...")
    create_advanced_visualizations(models, X_train, X_test, y_train, y_test, vis_dir)

def preprocess_data(file_path):
    if not os.path.exists(file_path):
        alternative_path = os.path.join('Data', os.path.basename(file_path))
        if os.path.exists(alternative_path):
            file_path = alternative_path
            print(f"File not found at {file_path}, using {alternative_path} instead")
        else:
            raise FileNotFoundError(f"Could not find file at {file_path} or {alternative_path}")
    
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    print(f"Dataset shape: {data.shape}")
    
    target_column = 'popularity_proxy'
    categorical_columns = ['platform', 'type', 'rating', 'genre_list_clean', 'genre_list', 'listed_in']
    id_columns = ['show_id', 'title']
    
    df_processed = data.copy()
    df_processed = df_processed.drop(id_columns, axis=1)
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    
    class_counts = y.value_counts()
    print(f"Number of classes: {len(class_counts)}")
    print(f"Smallest class count: {class_counts.min()}")
    print(f"Largest class count: {class_counts.max()}")
    
    if class_counts.min() < 2:
        print("Removing classes with less than 2 samples to allow stratification...")
        classes_to_keep = class_counts[class_counts >= 2].index
        mask = y.isin(classes_to_keep)
        X = X[mask]
        y = y[mask]
        print(f"After removing rare classes: {X.shape[0]} samples remaining")

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"Preprocessing complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y, X.columns

def train_models(X_train, y_train, X_test, y_test, model_dir):
    """Train the three models and save them"""
    models = []
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    joblib.dump(lr_model, os.path.join(model_dir, 'Logistic_Regression.pkl'))
    models.append(('Logistic Regression', lr_model))
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    joblib.dump(rf_model, os.path.join(model_dir, 'Random_Forest.pkl'))
    models.append(('Random Forest', rf_model))
    
    # XGBoost - need to relabel classes to be consecutive integers
    print("Training XGBoost...")
    xgb_label_encoder = LabelEncoder()
    y_train_xgb = xgb_label_encoder.fit_transform(y_train)
    y_test_xgb = xgb_label_encoder.transform(y_test)
    
    joblib.dump(xgb_label_encoder, os.path.join(model_dir, 'XGBoost_label_encoder.pkl'))
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_xgb)
    
    xgb_pred_encoded = xgb_model.predict(X_test)
    xgb_pred = xgb_label_encoder.inverse_transform(xgb_pred_encoded)
    
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    joblib.dump(xgb_model, os.path.join(model_dir, 'XGBoost.pkl'))
    models.append(('XGBoost', xgb_model))
    
    return models

def tune_hyperparameters(X_train, y_train, X_test, y_test, model_dir):
    """Perform hyperparameter tuning for the models"""
    tuned_models = []

    if X_train.shape[0] > 10000:
        print("Using a subset of data for hyperparameter tuning...")
        X_subset, _, y_subset, _ = train_test_split(
            X_train, y_train, train_size=10000, random_state=42, stratify=y_train
        )
    else:
        X_subset, y_subset = X_train, y_train
    
    print("Tuning Logistic Regression...")
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [1000]
    }
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_subset, y_subset)
    print(f"Best parameters: {lr_grid.best_params_}")
    
    best_lr = LogisticRegression(
        random_state=42,
        **lr_grid.best_params_
    )
    best_lr.fit(X_train, y_train)
    lr_pred = best_lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Tuned Logistic Regression Accuracy: {lr_accuracy:.4f}")
    joblib.dump(best_lr, os.path.join(model_dir, 'Logistic_Regression_tuned.pkl'))
    tuned_models.append(('Tuned Logistic Regression', best_lr))
    
    results = pd.DataFrame(lr_grid.cv_results_)
    important_params = list(lr_param_grid.keys())[:2]
    
    try:
        pivot_table = results.pivot_table(
            values='mean_test_score', 
            index=f'param_{important_params[0]}', 
            columns=f'param_{important_params[1]}'
        )
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
        plt.title(f'Hyperparameter Tuning Results for Logistic Regression')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, '../Visualizations/Model_Analysis/hyperparameter_tuning_Logistic_Regression.png'))
        plt.close()
    except:
        print("Could not create pivot table for Logistic Regression parameters")
    
    # Random Forest tuning
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X_subset, y_subset)
    print(f"Best parameters: {rf_grid.best_params_}")
    
    best_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **rf_grid.best_params_
    )
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Tuned Random Forest Accuracy: {rf_accuracy:.4f}")
    joblib.dump(best_rf, os.path.join(model_dir, 'Random_Forest_tuned.pkl'))
    tuned_models.append(('Tuned Random Forest', best_rf))
    
    # XGBoost tuning
    print("Tuning XGBoost...")
    xgb_label_encoder = LabelEncoder()
    y_subset_xgb = xgb_label_encoder.fit_transform(y_subset)
    y_train_xgb = xgb_label_encoder.transform(y_train)
    y_test_xgb = xgb_label_encoder.transform(y_test)

    joblib.dump(xgb_label_encoder, os.path.join(model_dir, 'XGBoost_tuned_label_encoder.pkl'))

    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(objective='multi:softmax', random_state=42),
        xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    xgb_grid.fit(X_subset, y_subset_xgb)
    print(f"Best parameters: {xgb_grid.best_params_}")
    
    best_xgb = xgb.XGBClassifier(
        objective='multi:softmax',
        random_state=42,
        n_jobs=-1,
        **xgb_grid.best_params_
    )
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"Tuned XGBoost Accuracy: {xgb_accuracy:.4f}")
    joblib.dump(best_xgb, os.path.join(model_dir, 'XGBoost_tuned.pkl'))
    tuned_models.append(('Tuned XGBoost', best_xgb))
    
    plt.figure(figsize=(10, 6))
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    accuracies = [lr_accuracy, rf_accuracy, xgb_accuracy]
    
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.title('Tuned Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, '../Visualizations/Model_Analysis/tuned_models_comparison.png'))
    plt.close()
    
    hyperparameter_summary = pd.DataFrame({
        'Logistic Regression': pd.Series(lr_grid.best_params_),
        'Random Forest': pd.Series(rf_grid.best_params_),
        'XGBoost': pd.Series(xgb_grid.best_params_)
    })
    
    hyperparameter_summary.to_csv(os.path.join(model_dir, '../Visualizations/Model_Analysis/best_hyperparameters.csv'))
    
    return tuned_models

def load_models(models_dir):
    models = []

    model_files = {
        'Logistic Regression': 'Logistic_Regression.pkl',
        'Random Forest': 'Random_Forest.pkl',
        'XGBoost': 'XGBoost.pkl'
    }

    tuned_model_files = {
        'Tuned Logistic Regression': 'Logistic_Regression_tuned.pkl',
        'Tuned Random Forest': 'Random_Forest_tuned.pkl',
        'Tuned XGBoost': 'XGBoost_tuned.pkl'
    }

    for name, file in model_files.items():
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                models.append((name, model))
                print(f"Loaded {name} model")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
    
    for name, file in tuned_model_files.items():
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                models.append((name, model))
                print(f"Loaded {name} model")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
    
    return models

def evaluate_models(models, X_test, y_test, feature_names, vis_dir, model_dir):
    all_metrics = {}
    all_predictions = {}
    
    for name, model in models:
        if "XGBoost" in name:
            encoder_path = os.path.join(
                model_dir, 
                'XGBoost_label_encoder.pkl' if 'tuned' not in name.lower() else 'XGBoost_tuned_label_encoder.pkl'
            )
            try:
                label_encoder = joblib.load(encoder_path)
                y_pred_encoded = model.predict(X_test)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            except Exception as e:
                print(f"Warning: Could not apply label encoding for {name}: {e}")
                y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        all_metrics[name] = metrics
        all_predictions[name] = y_pred
    
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    metrics_df.to_csv(os.path.join(vis_dir, 'model_metrics.csv'), index=False)
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='accuracy', data=metrics_df)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. All metrics comparison
    plt.figure(figsize=(14, 8))
    metrics_df.set_index('Model').plot(kind='bar', figsize=(14, 8))
    plt.title('Model Performance Metrics')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'all_metrics_comparison.png'))
    plt.close()
    
    # 3. Confusion matrices for each model
    for name, model in models:
        plt.figure(figsize=(10, 8))
        if "XGBoost" in name:
            encoder_path = os.path.join(
                model_dir, 
                'XGBoost_label_encoder.pkl' if 'tuned' not in name.lower() else 'XGBoost_tuned_label_encoder.pkl'
            )
            try:
                label_encoder = joblib.load(encoder_path)
                y_pred_encoded = model.predict(X_test)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            except Exception as e:
                print(f"Warning: Could not apply label encoding for {name} confusion matrix: {e}")
                y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
    
        y_test_grouped = y_test.copy()
        y_pred_grouped = y_pred.copy()
        top_classes = pd.Series(y_test).value_counts().nlargest(10).index
        
        y_test_grouped = np.where(np.isin(y_test, top_classes), y_test, -1)
        y_pred_grouped = np.where(np.isin(y_pred, top_classes), y_pred, -1)
        
        cm = confusion_matrix(y_test_grouped, y_pred_grouped)
        
        class_labels = [str(cls) for cls in top_classes] + ['Other']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, 
                    yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()
    
    # 4. Feature importance for tree-based models
    for name, model in models:
        if any(model_type in name for model_type in ['Random Forest', 'XGBoost']):
            plt.figure(figsize=(12, 10))
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                indices = np.argsort(importances)[::-1]
                
                top_n = min(20, len(feature_names))
                indices = indices[:top_n]
                plt.title(f'Top {top_n} Feature Importances - {name}')
                plt.barh(range(top_n), importances[indices], align='center')
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'feature_importance_{name.replace(" ", "_")}.png'))
                plt.close()

def create_advanced_visualizations(models, X_train, X_test, y_train, y_test, vis_dir):
    
    # 1. Create ROC curves
    create_roc_curves(models, X_test, y_test, vis_dir)
    
    # 2. Create precision-recall curves
    create_precision_recall_curves(models, X_test, y_test, vis_dir)
    
    
    # 3. Create class performance visualization
    create_class_performance_visualization(models, X_test, y_test, vis_dir)
    
    # 4. Create correlation visualization
    create_correlation_visualization(X_test, y_test, vis_dir)
    

def create_roc_curves(models, X_test, y_test, vis_dir):
    print("Creating ROC curves...")
    top_classes = pd.Series(y_test).value_counts().nlargest(5).index
    
    plt.figure(figsize=(12, 10))
    
    # For each model
    for name, model in models:
        for i, cls in enumerate(top_classes):
            y_test_bin = (y_test == cls).astype(int)
            try:
                y_score = model.predict_proba(X_test)[:, np.where(model.classes_ == cls)[0][0]]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{name} - Class {cls} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Could not calculate ROC for {name}, class {cls}: {e}")
    
    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Top 5 Classes')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(vis_dir, 'roc_curves.png'))
    plt.close()

def create_precision_recall_curves(models, X_test, y_test, vis_dir):
    print("Creating precision-recall curves...")
    top_classes = pd.Series(y_test).value_counts().nlargest(5).index
    
    plt.figure(figsize=(12, 10))
    
    for name, model in models:
        for i, cls in enumerate(top_classes):
            y_test_bin = (y_test == cls).astype(int)
            
            try:
                y_score = model.predict_proba(X_test)[:, np.where(model.classes_ == cls)[0][0]]
                
                precision, recall, _ = precision_recall_curve(y_test_bin, y_score)
                ap = average_precision_score(y_test_bin, y_score)
                
                plt.plot(recall, precision, lw=2, 
                        label=f'{name} - Class {cls} (AP = {ap:.2f})')
            except Exception as e:
                print(f"Could not calculate PR curve for {name}, class {cls}: {e}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Top 5 Classes')
    plt.legend(loc="lower left", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'precision_recall_curves.png'))
    plt.close()

def create_class_performance_visualization(models, X_test, y_test, vis_dir):
    print("Creating class performance visualizations...")
    
    model_predictions = {}
    for name, model in models:
        model_predictions[name] = model.predict(X_test)
    
    top_classes = pd.Series(y_test).value_counts().nlargest(10).index
    
    class_accuracies = {}
    for name, preds in model_predictions.items():
        class_accuracies[name] = {}
        for cls in top_classes:
            idx = y_test == cls
            if np.sum(idx) > 0:
                class_accuracies[name][cls] = np.mean(preds[idx] == y_test[idx])
    
    acc_data = []
    for model_name, class_acc in class_accuracies.items():
        for cls, acc in class_acc.items():
            acc_data.append({'Model': model_name, 'Class': str(cls), 'Accuracy': acc})
    
    acc_df = pd.DataFrame(acc_data)
    
    plt.figure(figsize=(14, 8))
    pivot_table = acc_df.pivot(index='Model', columns='Class', values='Accuracy')
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Model Accuracy by Class')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'class_performance_heatmap.png'))
    plt.close()

def create_correlation_visualization(X_test, y_test, vis_dir):
    print("Creating correlation visualization...")
    
    corr_data = X_test.copy()
    corr_data['target'] = y_test
    
    correlation = corr_data.corr()
    target_corr = correlation['target'].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = target_corr.drop('target').abs().nlargest(20).index
    
    top_corr = target_corr[top_features]
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top 20 Feature Correlations with Target')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_target_correlation.png'))
    plt.close()
    
    plt.figure(figsize=(14, 12))
    top_corr_matrix = correlation.loc[top_features, top_features]
    sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_correlation_matrix.png'))
    plt.close()

if __name__ == "__main__":
    main()