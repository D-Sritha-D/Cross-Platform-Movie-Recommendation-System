import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score, 
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate and visualize recommendation models')
    parser.add_argument('--data', type=str, default='Data/selected_features_for_recommendation.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for models and visualizations')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run evaluation on existing models')
    args = parser.parse_args()
    
    # Create output directories
    model_dir = os.path.join(args.output, 'Models')
    vis_dir = os.path.join(args.output, 'Visualizations/Model_Analysis')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nModel Training and Evaluation Pipeline")
    print(f"==============================================")
    print(f"Data file: {args.data}")
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
    else:
        print("\nSkipping training, loading existing models...")
        models = load_models(model_dir)
        if not models:
            print("No models found")
            return
    
    # Step 3: Model evaluation
    print("\nStep 3: Evaluating models and creating visualizations...")
    evaluate_models(models, X_test, y_test, feature_names, vis_dir, model_dir)
    
    # Step 4: Advanced visualizations
    print("\nStep 4: Creating advanced visualizations...")
    create_advanced_visualizations(models, X_test, y_test, vis_dir, model_dir)
    
    print("\nPipeline completed successfully!")
    print(f"Models saved to: {model_dir}")
    print(f"Visualizations saved to: {vis_dir}")

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
    """Train models with overfitting prevention and enhanced tracking"""
    import time
    import numpy as np
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import matplotlib.pyplot as plt
    models = []

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        solver='saga',
        C=0.5,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        penalty='elasticnet',
        l1_ratio=0.5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    print(f"Completed in {time.time()-start_time:.1f}s - Iterations: {lr_model.n_iter_[0]}")
    
    lr_pred = lr_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    joblib.dump(lr_model, os.path.join(model_dir, 'Logistic_Regression.pkl'))
    models.append(('Logistic Regression', lr_model))

    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,  # Reduced from 200
        max_depth=8,       # Reduced from 10
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    print(f"Completed in {time.time()-start_time:.1f}s")
    
    rf_pred = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    joblib.dump(rf_model, os.path.join(model_dir, 'Random_Forest.pkl'))
    models.append(('Random Forest', rf_model))

    # 3. XGBoost with Anti-Overfitting Measures
    print("\nTraining XGBoost (with overfitting controls)...")
    xgb_label_encoder = LabelEncoder()
    y_train_xgb = xgb_label_encoder.fit_transform(y_train)
    y_test_xgb = xgb_label_encoder.transform(y_test)
    
    # Calculate balanced class weights
    class_counts = np.bincount(y_train_xgb)
    weight_ratio = np.sum(class_counts) / (len(class_counts) * class_counts)
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=500, 
        learning_rate=0.01,  
        max_depth=3,       
        min_child_weight=5, 
        gamma=0.2,        
        subsample=0.7,   
        colsample_bytree=0.7, 
        reg_alpha=0.5,     
        reg_lambda=1.5,    
        scale_pos_weight=weight_ratio,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15,  
        eval_metric='mlogloss'
    )
    
    # Create validation set
    X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(
        X_train, y_train_xgb, test_size=0.2, random_state=42)  
    
    print("Training with early stopping...")
    start_time = time.time()
    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_train_xgb, y_train_xgb), (X_val, y_val)],
        verbose=10
    )
    
    # Plot training history
    results = xgb_model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['mlogloss'], label='Train')
    plt.plot(results['validation_1']['mlogloss'], label='Validation')
    plt.title('XGBoost Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, '../Visualizations/Model_Analysis/xgboost_training.png'))
    plt.close()
    
    print(f"Completed in {time.time()-start_time:.1f}s")
    print(f"Best iteration: {xgb_model.best_iteration+1}/{xgb_model.n_estimators}")
    
    xgb_pred = xgb_label_encoder.inverse_transform(xgb_model.predict(X_test))
    print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
    
    # Feature importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(xgb_model, max_num_features=20)
    plt.savefig(os.path.join(model_dir, '../Visualizations/Model_Analysis/xgboost_importance.png'))
    plt.close()
    
    joblib.dump(xgb_model, os.path.join(model_dir, 'XGBoost.pkl'))
    joblib.dump(xgb_label_encoder, os.path.join(model_dir, 'XGBoost_label_encoder.pkl'))
    models.append(('XGBoost', xgb_model))
    
    return models

def load_models(models_dir):
    """Load saved models"""
    models = []
    model_files = {
        'Logistic Regression': 'Logistic_Regression.pkl',
        'Random Forest': 'Random_Forest.pkl',
        'XGBoost': 'XGBoost.pkl'
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
    
    return models

def evaluate_models(models, X_test, y_test, feature_names, vis_dir, model_dir):
    all_metrics = {}
    all_predictions = {}
    
    for name, model in models:
        if "XGBoost" in name:
            encoder_path = os.path.join(model_dir, 'XGBoost_label_encoder.pkl')
            try:
                label_encoder = joblib.load(encoder_path)
                y_pred_encoded = model.predict(X_test)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
                y_proba = model.predict_proba(X_test)
            except Exception as e:
                print(f"Warning: Could not apply label encoding for {name}: {e}")
                y_pred = model.predict(X_test)
                y_proba = None
        else:
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
        
        # Enhanced metrics calculation
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': None,
            'top_class_accuracy': None
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(
                    label_binarize(y_test, classes=np.unique(y_test)),
                    y_proba,
                    multi_class='ovr',
                    average='weighted'
                )
            except:
                pass
        
        # Calculate top class accuracy
        top_class = y_test.value_counts().index[0]
        top_mask = y_test == top_class
        metrics['top_class_accuracy'] = accuracy_score(
            y_test[top_mask], y_pred[top_mask])
        
        all_metrics[name] = metrics
        all_predictions[name] = y_pred
    
    # Save enhanced metrics
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    metrics_df.to_csv(os.path.join(vis_dir, 'model_metrics.csv'), index=False)
    
    # Enhanced visualization
    plt.figure(figsize=(14, 8))
    metrics_df.set_index('Model').plot(kind='bar', figsize=(14, 8))
    plt.title('Enhanced Model Performance Metrics')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'metrics_comparison.png'))
    plt.close()
    

def create_advanced_visualizations(models, X_test, y_test, vis_dir, model_dir):
    """Create all advanced visualizations including confusion matrices"""
    create_roc_curves(models, X_test, y_test, vis_dir)
    create_correlation_visualization(X_test, y_test, vis_dir)
    
    # Add confusion matrix plotting
    print("\nCreating confusion matrices...")
    for name, model in models:
        plt.figure(figsize=(10, 8))
        if "XGBoost" in name:
            encoder_path = os.path.join(model_dir, 'XGBoost_label_encoder.pkl')
            try:
                label_encoder = joblib.load(encoder_path)
                y_pred_encoded = model.predict(X_test)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            except Exception as e:
                print(f"Warning: Could not apply label encoding for {name}: {e}")
                y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Get top 10 classes
        top_classes = pd.Series(y_test).value_counts().nlargest(10).index
        
        # Group remaining classes as 'Other'
        y_test_grouped = np.where(np.isin(y_test, top_classes), y_test, 'Other')
        y_pred_grouped = np.where(np.isin(y_pred, top_classes), y_pred, 'Other')
        
        cm = confusion_matrix(y_test_grouped, y_pred_grouped, labels=list(top_classes)+['Other'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(top_classes)+['Other'],
                    yticklabels=list(top_classes)+['Other'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()

def create_roc_curves(models, X_test, y_test, vis_dir):
    print("Creating ROC curves...")
    top_classes = pd.Series(y_test).value_counts().nlargest(5).index
    
    plt.figure(figsize=(14, 10))

    model_attributes = {
        'Logistic Regression': {'color': 'blue'},
        'Random Forest': {'color': 'green'},
        'XGBoost': {'color': 'red'}
    }
    
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    
    for name, model in models:
        for i, cls in enumerate(top_classes):
            y_test_bin = (y_test == cls).astype(int)
            
            try:
                y_score = model.predict_proba(X_test)[:, np.where(model.classes_ == cls)[0][0]]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                roc_auc = auc(fpr, tpr)

                display_name = model_attributes[name]['display_name']
                color = model_attributes[name]['color']
                
                plt.plot(fpr, tpr, 
                        color=color,
                        linestyle=line_styles[i % len(line_styles)],
                        lw=2.5,
                        label=f'{display_name} - Class {cls} (AUC = {roc_auc:.2f})')
                
            except Exception as e:
                print(f"Could not calculate ROC for {name}, class {cls}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Top 5 Classes', fontsize=14)
    
    # Create clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc="lower right", 
              fontsize=10,
              framealpha=1)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'roc_curves.png'), dpi=300)
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