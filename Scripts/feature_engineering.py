import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import ast
import os

def load_data(filepath="Data/all_platforms_combined.csv"):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def extract_genres(genre_string):
    if pd.isna(genre_string):
        return []
    try:
        return ast.literal_eval(genre_string)
    except (ValueError, SyntaxError):
        return [g.strip() for g in str(genre_string).split(',') if g.strip()]

def extract_features(df):
    df_features = df.copy()
    
    # 1. Basic cleaning
    for col in ['description', 'director', 'cast', 'country', 'listed_in']:
        df_features[col] = df_features[col].fillna('')
    
    # 2. Content age feature
    current_year = 2024  # Updated to current year
    df_features['content_age'] = current_year - df_features['release_year']
    
    # 3. Extract release decade
    df_features['release_decade'] = (df_features['release_year'] // 10) * 10
    
    # 4. Genre count feature - use existing genre_list without creating new column
    df_features['genre_count'] = df_features['genre_list'].apply(
        lambda x: len(extract_genres(x)) if pd.notna(x) else 0)
    
    # 5. Process director features
    df_features['director_count'] = df_features['director'].apply(
        lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(',')))
    
    # 6. Cast size feature
    df_features['cast_size'] = df_features['cast'].apply(
        lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(',')))
    
    # 7. Content type features (Movie vs TV Show)
    df_features['is_movie'] = (df_features['type'] == 'Movie').astype(int)
    
    # 8. Country count feature
    df_features['country_count'] = df_features['country'].apply(
        lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(',')))
    
    # 9. Is US content
    df_features['is_us'] = df_features['country'].str.contains(
        'United States|USA|US', case=False, na=False).astype(int)
    
    # 10. Content duration features
    if 'duration_numeric' in df_features.columns:
        # Normalize duration differently for movies vs TV shows
        movie_mask = df_features['duration_unit'] == 'min'
        tv_mask = df_features['duration_unit'] == 'season'
        
        # For movies
        movie_durations = df_features.loc[movie_mask, 'duration_numeric']
        movie_mean = movie_durations.mean()
        movie_std = movie_durations.std()
        df_features.loc[movie_mask, 'duration_normalized'] = (movie_durations - movie_mean) / movie_std
        
        # For TV shows
        tv_durations = df_features.loc[tv_mask, 'duration_numeric']
        tv_mean = tv_durations.mean()
        tv_std = tv_durations.std()
        df_features.loc[tv_mask, 'duration_normalized'] = (tv_durations - tv_mean) / tv_std
        
        df_features['duration_normalized'] = df_features['duration_normalized'].fillna(0)
    
    # 11. Rating category
    rating_groups = {
        'Kids': ['TV-Y', 'TV-Y7', 'G', 'TV-G'],
        'Family': ['PG', 'TV-PG'],
        'Teen': ['PG-13', 'TV-14'],
        'Adult': ['R', 'TV-MA', 'NC-17'],
        'Unknown': ['NR', 'UR', 'NOT RATED', 'UNRATED', 'N/A', 'NA']
    }
    
    df_features['rating_category'] = 'Unknown'
    for category, ratings in rating_groups.items():
        mask = df_features['rating'].isin(ratings)
        df_features.loc[mask, 'rating_category'] = category
    
    # 12. Description length
    df_features['description_length'] = df_features['description'].str.len()
    
    # 13. Title length
    df_features['title_length'] = df_features['title'].str.len()
    
    # 14. Platform as a feature (not a target)
    platform_encoder = LabelEncoder()
    df_features['platform_encoded'] = platform_encoder.fit_transform(df_features['platform'])
    
    # 15. Genre embeddings (using show_id as a key for lookup tables)
    genre_data = {}
    for _, row in df_features.iterrows():
        show_id = row['show_id']
        genres = extract_genres(row['genre_list'])
        genre_data[show_id] = genres
    
    df_features['genre_data'] = df_features['show_id'].map(genre_data)
    
    return df_features

# Process textual features using TF-IDF and dimensionality reduction
def prepare_textual_features(df, text_column='description', n_components=100):
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        stop_words='english'
    )
    
    tfidf_matrix = tfidf.fit_transform(df[text_column].fillna(''))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_features = svd.fit_transform(tfidf_matrix)
    
    feature_names = [f'{text_column}_svd_{i}' for i in range(n_components)]

    text_df = pd.DataFrame(text_features, columns=feature_names)

    return text_df, tfidf, svd

def create_genre_features(df, top_n=20):
    all_genres = []
    for genre_list in df['genre_list']:
        genres = extract_genres(genre_list)
        all_genres.extend(genres)
    
    top_genres = [genre for genre, _ in Counter(all_genres).most_common(top_n)]
    genre_features = pd.DataFrame(index=df.index)
    for genre in top_genres:
        genre_features[f'genre_{genre.lower().replace(" ", "_")}'] = df['genre_list'].apply(
            lambda x: 1 if genre in extract_genres(x) else 0)
    
    return genre_features

def select_features(df_features, text_features, genre_features, k=100):
    """
    Select top features for similarity-based recommendations
    instead of platform classification
    """
    X = pd.concat([
        df_features.select_dtypes(include=['number']),
        pd.get_dummies(df_features[['rating_category']], drop_first=True),
        text_features,
        genre_features
    ], axis=1)
    
    X = X.drop(columns=['platform_encoded'], errors='ignore')
    
    X = X.fillna(0)
    
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    X_filtered = selector.fit_transform(X)
    
    variances = selector.variances_
    feature_variance = pd.DataFrame({
        'feature': X.columns,
        'variance': variances
    }).sort_values('variance', ascending=False)
    
    top_features = feature_variance.head(k)['feature'].tolist()
    X_selected = X[top_features]
    
    return X_selected, top_features

def get_similarity_features(df_features, text_features, genre_features):

    content_features = [
        'release_year', 'content_age', 'duration_normalized', 
        'genre_count', 'description_length', 'is_movie'
    ]

    available_features = [f for f in content_features if f in df_features.columns]
    X_numerical = df_features[available_features].fillna(0)

    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)

    X_combined = np.hstack([
        X_numerical_scaled,
        text_features.values,
        genre_features.values
    ])

    return X_combined, scaler

def prepare_content_similarity_dataset(df_features, text_features, genre_features, selected_features):

    X = pd.concat([
        df_features.select_dtypes(include=['number']),
        pd.get_dummies(df_features[['rating_category']], drop_first=True),
        text_features,
        genre_features
    ], axis=1)
    
    available_features = [f for f in selected_features if f in X.columns]
    X_final = X[available_features]

    reference_df = df_features[['show_id', 'title', 'platform', 'type', 'rating_category', 'genre_list']].copy()
    result = pd.concat([reference_df, X_final], axis=1)
    
    return result

def save_recommendation_dataset(recommendation_df, output_dir="Data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    recommendation_df.to_csv(f"{output_dir}/recommendation_features.csv", index=False)
    print(f"Recommendation dataset saved to {output_dir}/recommendation_features.csv")

def run_feature_engineering_pipeline(data_path="Data/all_platforms_combined.csv"):
    print("Loading data...")
    df = load_data(data_path)

    print("Extracting features...")
    df_features = extract_features(df)

    print("Preparing textual features...")
    text_features, tfidf_vectorizer, svd = prepare_textual_features(df_features)

    print("Creating genre features...")
    genre_features = create_genre_features(df_features)

    print("Selecting important features...")
    X_selected, selected_features = select_features(df_features, text_features, genre_features)

    print("Preparing similarity features...")
    X_similarity, scaler = get_similarity_features(df_features, text_features, genre_features)

    print("Creating recommendation dataset...")
    recommendation_df = prepare_content_similarity_dataset(
        df_features, text_features, genre_features, selected_features)

    print("Saving engineered features...")
    df_features.to_csv("Data/engineered_features.csv", index=False)

    save_recommendation_dataset(recommendation_df)
    
    print(f"Feature engineering complete. Total items: {len(recommendation_df)}")

    platform_counts = recommendation_df['platform'].value_counts()
    print("\nPlatform distribution:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count} items ({count/len(recommendation_df)*100:.1f}%)")

    return {
        'df_features': df_features,
        'text_features': text_features,
        'genre_features': genre_features,
        'X_selected': X_selected,
        'selected_features': selected_features,
        'X_similarity': X_similarity,
        'scaler': scaler,
        'tfidf_vectorizer': tfidf_vectorizer,
        'svd': svd,
        'recommendation_df': recommendation_df
    }

if __name__ == "__main__":
    results = run_feature_engineering_pipeline()

    print("\nDataset shapes:")
    print(f"Recommendation features: {results['recommendation_df'].shape}")
    print(f"Selected features: {len(results['selected_features'])}")