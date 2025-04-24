import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.makedirs('Visualizations/Feature_Analysis', exist_ok=True)
# Load the data
df = pd.read_csv("Data/all_platforms_combined.csv")

# Basic data exploration
print("Data shape:", df.shape)
print("\nFeature data types:")
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    if col in ['title', 'director', 'cast', 'description']:
        continue
    
    le = LabelEncoder()
    if df[col].isna().any():
        df_encoded[col] = df_encoded[col].fillna('unknown')
    
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df[col].isna().any():
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())

# Feature importance using Random Forest
target = 'platform'
if target in df_encoded.columns:
    X = df_encoded.drop([target, 'title', 'director', 'cast', 'description'], axis=1, errors='ignore')
    y = df_encoded[target]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature importance using Random Forest:")
    print(feature_importance)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for ' + target)
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Analysis/feature_importance.png')

# Mutual Information for numerical features
def calculate_mi(df, target_col):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if not numerical_cols:
        return None
    
    mi_scores = mutual_info_regression(df[numerical_cols], df[target_col])
    mi_df = pd.DataFrame({
        'Feature': numerical_cols,
        'MI_Score': mi_scores
    }).sort_values(by='MI_Score', ascending=False)
    
    return mi_df

if 'release_year' in df_encoded.columns:
    mi_year = calculate_mi(df_encoded, 'release_year')
    if mi_year is not None:
        print("\nMutual Information scores with release_year:")
        print(mi_year)

correlation_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
if len(correlation_cols) > 1:
    correlation_matrix = df_encoded[correlation_cols].corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Analysis/correlation_matrix.png')

# PCA to identify most informative features
if len(correlation_cols) > 2:
    X_numeric = df_encoded[correlation_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA explained variance ratio:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.4f}")
    
    print("\nCumulative explained variance:")
    for i, var in enumerate(cumulative_variance):
        print(f"PC1-PC{i+1}: {var:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Analysis/pca_explained_variance.png')
    
    components = pd.DataFrame(pca.components_, columns=correlation_cols)
    print("\nPCA feature loadings for top components:")
    print(components.iloc[:3])  # Show top 3 components

# Identify most relevant features based on all analyses
print("\n=== Most Relevant Features ===")
print("Based on the analyses, the most relevant features are:")

if 'feature_importance' in locals():
    print("\nTop 5 features from Random Forest:")
    print(feature_importance.head(5))

if 'mi_year' in locals() and mi_year is not None:
    print("\nTop 5 features from Mutual Information:")
    print(mi_year.head(5))

print("\nSummary of findings:")
print("1. The features with highest Random Forest importance scores")
print("2. Features with high mutual information scores")
print("3. Features with significant PCA loadings in the first few components")
print("4. Features with strong correlations to other important features")