import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
os.makedirs('Visualizations/Feature_Selection', exist_ok=True)

print("Loading processed data...")
df = pd.read_csv('Data/processed_streaming_data.csv')

# Function for feature selection
def select_best_features(df):
    print("Starting feature selection process...")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    columns_to_exclude = ['show_id', 'title', 'director', 'cast', 'country', 'date_added', 
                         'description', 'date_added_clean', 'genre_list', 'text_content',
                         'genre_list_clean', 'day_name', 'month_name', 'release_period']
    
    numerical_features = [col for col in numerical_features if col not in columns_to_exclude]
    categorical_features = [col for col in categorical_features if col not in columns_to_exclude]
    
    feature_df = df.copy()
    for col in numerical_features:
        if feature_df[col].isnull().sum() > 0:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    le = LabelEncoder()
    for col in categorical_features:
        if feature_df[col].isnull().sum() > 0:
            feature_df[col] = feature_df[col].fillna('Unknown')
        feature_df[col + '_encoded'] = le.fit_transform(feature_df[col])
    
    encoded_categorical = [col + '_encoded' for col in categorical_features]
    
    all_features = numerical_features + encoded_categorical
    
    # Method 1: Feature Importance using Random Forest
    print("Calculating Random Forest feature importance...")
    X = feature_df[all_features]
    y = feature_df['platform'] 
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)

    top_rf_features = rf_importance.head(20)['Feature'].tolist()
    
    # Method 2: Mutual Information
    print("Calculating Mutual Information scores...")
    
    X_mi = feature_df[all_features]
    y_mi = feature_df['release_year']
    
    mi_selector = SelectKBest(mutual_info_regression, k='all')
    mi_selector.fit(X_mi, y_mi)

    mi_scores = pd.DataFrame({
        'Feature': all_features,
        'MI_Score': mi_selector.scores_
    }).sort_values(by='MI_Score', ascending=False)
    
    top_mi_features = mi_scores.head(20)['Feature'].tolist()
    
    # Method 3: Correlation Analysis
    print("Performing correlation analysis...")

    correlation_matrix = feature_df[numerical_features].corr()
    
    mean_correlation = pd.DataFrame({
        'Feature': numerical_features,
        'Mean_Correlation': correlation_matrix.abs().mean()
    }).sort_values(by='Mean_Correlation', ascending=False)

    correlated_features = mean_correlation.head(15)['Feature'].tolist()
    
    # Method 4: PCA
    print("Performing PCA for feature selection...")
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(feature_df[numerical_features])
    pca = PCA()
    pca.fit(X_pca)
    
    loading_scores = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=numerical_features
    )

    abs_loading_scores = loading_scores.abs()
    
    pca_selected_features = []
    for i in range(3):
        component_features = abs_loading_scores.nlargest(5, f'PC{i+1}').index.tolist()
        pca_selected_features.extend(component_features)
    
    pca_selected_features = list(set(pca_selected_features))
    
    print("Combining results from different feature selection methods...")
    
    all_selected_features = top_rf_features + top_mi_features + correlated_features + pca_selected_features
    feature_counts = pd.Series(all_selected_features).value_counts().reset_index()
    feature_counts.columns = ['Feature', 'Selection_Count']
    feature_counts = feature_counts.sort_values(['Selection_Count', 'Feature'], ascending=[False, True])
    
    top_features = feature_counts[feature_counts['Selection_Count'] >= 2]['Feature'].tolist()
    
    if len(top_features) < 15:
        remaining = 15 - len(top_features)
        additional_features = feature_counts[feature_counts['Selection_Count'] == 1]['Feature'].head(remaining).tolist()
        top_features.extend(additional_features)
    
    numerical_selected = [f for f in top_features if f in numerical_features]
    categorical_selected = [f.replace('_encoded', '') for f in top_features if f in encoded_categorical]
    genre_features = [col for col in df.columns if col.startswith('genre_') and col != 'genre_count']
    
    must_include = ['platform', 'type', 'duration_numeric', 'content_age', 'min_age', 
                   'genre_count', 'recency_score', 'popularity_proxy']
    
    for feature in must_include:
        if feature not in numerical_selected and feature not in categorical_selected:
            if feature in numerical_features:
                numerical_selected.append(feature)
            elif feature in categorical_features:
                categorical_selected.append(feature)

    tfidf_features = [col for col in df.columns if col.startswith('tfidf_pca_')]
    
    
    final_features = list(set(numerical_selected + categorical_selected + genre_features + tfidf_features))
    
    create_visualizations(feature_df, rf_importance, mi_scores, correlation_matrix, pca, feature_counts, final_features)
    
    return final_features, numerical_selected, categorical_selected, genre_features, tfidf_features

def create_visualizations(df, rf_importance, mi_scores, correlation_matrix, pca, feature_counts, final_features):
    print("Creating visualizations for feature selection...")
    
    # 1. Random Forest Feature Importance
    plt.figure(figsize=(12, 10))
    top_rf = rf_importance.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_rf)
    plt.title('Top 20 Features by Random Forest Importance')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/rf_importance.png')
    
    # 2. Mutual Information Scores
    plt.figure(figsize=(12, 10))
    top_mi = mi_scores.head(20)
    sns.barplot(x='MI_Score', y='Feature', data=top_mi)
    plt.title('Top 20 Features by Mutual Information Score')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/mutual_information.png')
    
    # 3. Correlation Heatmap of numerical features
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                linewidths=0.5, vmax=1.0, vmin=-1.0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/correlation_heatmap.png')
    
    # 4. PCA Explained Variance
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/pca_variance.png')
    
    # 5. Feature Selection Frequency
    plt.figure(figsize=(14, 10))
    feature_counts_plot = feature_counts.head(30)
    sns.barplot(x='Selection_Count', y='Feature', data=feature_counts_plot)
    plt.title('Top 30 Features by Selection Frequency Across Methods')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/feature_selection_frequency.png')
    
    # 6. Final Selected Features
    plt.figure(figsize=(14, 10))
    final_features_df = pd.DataFrame({'Feature': final_features})
    final_features_df['Is_Selected'] = 1
    sns.barplot(x='Is_Selected', y='Feature', data=final_features_df)
    plt.title('Final Selected Features for Recommendation System')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Selection/final_features.png')

final_features, numerical_selected, categorical_selected, genre_features, tfidf_features = select_best_features(df)

selected_df = df[['show_id', 'title'] + final_features].copy()
selected_df.to_csv('selected_features_for_recommendation.csv', index=False)

print("\nFeature selection completed!")
print(f"Selected {len(final_features)} features for recommendation system")
print("\nSelected numerical features:", numerical_selected)
print("\nSelected categorical features:", categorical_selected)
print("\nSelected genre features:", len(genre_features), "genre features")
print("\nSelected text features:", tfidf_features)