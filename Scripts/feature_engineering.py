import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import re
from collections import Counter
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

os.makedirs('Visualizations/Feature_Engineering', exist_ok=True)

# Load the data
df = pd.read_csv("Data/all_platforms_combined.csv")

# Basic data cleaning
def clean_data(df):
    df_clean = df.copy()
    
    df_clean['duration_numeric'] = df_clean['duration_numeric'].fillna(df_clean['duration_numeric'].median())
    
    df_clean['date_added_clean'] = pd.to_datetime(df_clean['date_added'], errors='coerce')
    
    date_features = df_clean[~df_clean['date_added_clean'].isna()].copy()
    date_features['month_added'] = date_features['date_added_clean'].dt.month
    date_features['year_added'] = date_features['date_added_clean'].dt.year
    date_features['day_added'] = date_features['date_added_clean'].dt.day
    date_features['day_of_week_added'] = date_features['date_added_clean'].dt.dayofweek
    date_features['quarter_added'] = date_features['date_added_clean'].dt.quarter

    df_clean = df_clean.merge(date_features[['show_id', 'month_added', 'year_added', 'day_added', 
                                            'day_of_week_added', 'quarter_added']], 
                              on='show_id', how='left')
    
    # Calculate content age
    current_year = datetime.now().year
    df_clean['content_age'] = current_year - df_clean['release_year']

    # Create age category based on rating
    rating_map = {
        'TV-Y': 0,
        'TV-Y7': 7,
        'TV-G': 0,
        'TV-PG': 10,
        'TV-14': 14,
        'TV-MA': 18,
        'G': 0,
        'PG': 10,
        'PG-13': 13,
        'R': 17,
        'NC-17': 18,
        'NR': np.nan,
        'UR': np.nan
    }
    df_clean['min_age'] = df_clean['rating'].map(rating_map)
    df_clean['min_age'] = df_clean['min_age'].fillna(df_clean['min_age'].median())
    
    # Create age categories
    df_clean['age_category'] = pd.cut(
        df_clean['min_age'], 
        bins=[0, 7, 13, 16, 100], 
        labels=['Kids', 'Family', 'Teen', 'Adult']
    )
    
    # Create decade feature
    df_clean['decade'] = (df_clean['release_year'] // 10) * 10
    
    return df_clean

df_processed = clean_data(df)

# Process genre_list feature
def process_genres(df):
    df_genres = df.copy()
    
    df_genres['genre_list_clean'] = df_genres['genre_list'].fillna('[]')
    df_genres['genre_list_clean'] = df_genres['genre_list_clean'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    
    df_genres['genre_count'] = df_genres['genre_list_clean'].apply(len)
    
    all_genres = [genre for sublist in df_genres['genre_list_clean'] for genre in sublist]
    top_genres = [x[0] for x in Counter(all_genres).most_common(10)]
    
    for genre in top_genres:
        df_genres[f'genre_{genre}'] = df_genres['genre_list_clean'].apply(lambda x: 1 if genre in x else 0)
    
    return df_genres, top_genres

df_processed, top_genres = process_genres(df_processed)

# Text-based features from title and description
def create_text_features(df):
    df_text = df.copy()
    
    df_text['text_content'] = df_text['title'].fillna('') + ' ' + df_text['description'].fillna('')
    
    df_text['title_length'] = df_text['title'].fillna('').apply(len)
    
    df_text['description_length'] = df_text['description'].fillna('').apply(len)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_text['text_content'].fillna(''))
    
    pca = PCA(n_components=5)
    tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
    
    for i in range(5):
        df_text[f'tfidf_pca_{i}'] = tfidf_pca[:, i]
    
    return df_text

df_processed = create_text_features(df_processed)

# Cast and crew features
def create_cast_crew_features(df):
    df_cast = df.copy()
    
    df_cast['director_count'] = df_cast['director'].fillna('').apply(
        lambda x: len(x.split(',')) if x != '' else 0
    )
    
    df_cast['cast_count'] = df_cast['cast'].fillna('').apply(
        lambda x: len(x.split(',')) if x != '' else 0
    )
    
    df_cast['has_director'] = df_cast['director'].notna() & (df_cast['director'] != '')
    df_cast['has_director'] = df_cast['has_director'].astype(int)
    
    df_cast['has_cast'] = df_cast['cast'].notna() & (df_cast['cast'] != '')
    df_cast['has_cast'] = df_cast['has_cast'].astype(int)
    
    directors = df_cast['director'].fillna('').str.split(',').explode()
    directors = directors[directors != ''].str.strip()
    top_directors = [x[0] for x in Counter(directors).most_common(10)]
    
    cast = df_cast['cast'].fillna('').str.split(',').explode()
    cast = cast[cast != ''].str.strip()
    top_cast = [x[0] for x in Counter(cast).most_common(10)]
    
    for director in top_directors:
        df_cast[f'director_{director.replace(" ", "_")}'] = df_cast['director'].fillna('').apply(
            lambda x: 1 if director in x else 0
        )
    
    for actor in top_cast:
        df_cast[f'actor_{actor.replace(" ", "_")}'] = df_cast['cast'].fillna('').apply(
            lambda x: 1 if actor in x else 0
        )
    
    return df_cast, top_directors, top_cast

df_processed, top_directors, top_cast = create_cast_crew_features(df_processed)

# Interaction features between existing features
def create_interaction_features(df):
    df_inter = df.copy()
    
    max_age = df_inter['content_age'].max()
    df_inter['recency_score'] = 1 - (df_inter['content_age'] / max_age)
    
    for genre in top_genres:
        for platform in df_inter['platform'].unique():
            df_inter[f'{genre}_{platform}'] = ((df_inter[f'genre_{genre}'] == 1) & 
                                              (df_inter['platform'] == platform)).astype(int)
    
    df_inter['movie_duration'] = ((df_inter['type'] == 'Movie') * 
                                 df_inter['duration_numeric']).fillna(0)
    
    df_inter['show_duration'] = ((df_inter['type'] == 'TV Show') * 
                                df_inter['duration_numeric']).fillna(0)
    
    df_inter['release_period'] = df_inter['decade'].astype(str) + '_' + df_inter['age_category'].astype(str)
    
    df_inter['popularity_proxy'] = df_inter['cast_count'] + df_inter['genre_count']
    
    return df_inter

df_processed = create_interaction_features(df_processed)

df_processed.to_csv('processed_streaming_data.csv', index=False)

plt.figure(figsize=(20, 16))
plt.subplots_adjust(hspace=0.5)

# 1. Content age distribution by platform
plt.subplot(3, 3, 1)
sns.boxplot(x='platform', y='content_age', data=df_processed)
plt.title('Content Age Distribution by Platform')
plt.xticks(rotation=45)

# 2. Distribution of content by age category
plt.subplot(3, 3, 2)
age_category_counts = df_processed['age_category'].value_counts()
plt.pie(age_category_counts, labels=age_category_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Content by Age Category')

# 3. Genre distribution across platforms
plt.subplot(3, 3, 3)
genre_platform = df_processed.groupby(['platform'])[['genre_' + g for g in top_genres]].sum()
sns.heatmap(genre_platform, cmap='YlGnBu', annot=False)
plt.title('Genre Distribution Across Platforms')

# 4. Content added by year
plt.subplot(3, 3, 4)
sns.countplot(x='year_added', data=df_processed[~df_processed['year_added'].isna()].sort_values('year_added'))
plt.title('Content Added by Year')
plt.xticks(rotation=45)

# 5. Average content age by platform
plt.subplot(3, 3, 5)
platform_age = df_processed.groupby('platform')['content_age'].mean().sort_values(ascending=False)
sns.barplot(x=platform_age.index, y=platform_age.values)
plt.title('Average Content Age by Platform')
plt.xticks(rotation=45)

# 6. TF-IDF PCA components correlation
plt.subplot(3, 3, 6)
tfidf_cols = [col for col in df_processed.columns if col.startswith('tfidf_pca')]
tfidf_corr = df_processed[tfidf_cols].corr()
sns.heatmap(tfidf_corr, annot=True, cmap='coolwarm')
plt.title('TF-IDF PCA Components Correlation')

# 7. Genre count distribution
plt.subplot(3, 3, 7)
sns.histplot(df_processed['genre_count'], kde=True)
plt.title('Distribution of Genre Count')

# 8. Top 10 genres distribution
plt.subplot(3, 3, 8)
genre_counts = df_processed[[f'genre_{g}' for g in top_genres]].sum()
genre_counts.index = top_genres
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Top 10 Genres Distribution')
plt.xticks(rotation=90)

# 9. Content type distribution by platform
plt.subplot(3, 3, 9)
type_platform = pd.crosstab(df_processed['platform'], df_processed['type'])
type_platform_pct = type_platform.div(type_platform.sum(axis=1), axis=0)
type_platform_pct.plot(kind='bar', stacked=True)
plt.title('Content Type Distribution by Platform')
plt.xticks(rotation=45)

plt.savefig('Visualizations/Feature_Engineering/feature_visualizations_1.png', bbox_inches='tight')
plt.figure(figsize=(20, 16))
plt.subplots_adjust(hspace=0.5)

# 10. Movies vs TV Shows duration distribution
plt.subplot(3, 3, 1)
movie_durations = df_processed[df_processed['type'] == 'Movie']['duration_numeric']
show_durations = df_processed[df_processed['type'] == 'TV Show']['duration_numeric']
plt.hist([movie_durations, show_durations], bins=20, alpha=0.5, label=['Movies', 'TV Shows'])
plt.legend()
plt.title('Duration Distribution: Movies vs TV Shows')

# 11. Content released by decade
plt.subplot(3, 3, 2)
decade_counts = df_processed['decade'].value_counts().sort_index()
sns.barplot(x=decade_counts.index, y=decade_counts.values)
plt.title('Content Released by Decade')
plt.xticks(rotation=45)

# 12. Correlation matrix of key numerical features
plt.subplot(3, 3, 3)
num_features = ['content_age', 'duration_numeric', 'genre_count', 'cast_count', 
                'director_count', 'min_age', 'release_year']
corr = df_processed[num_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation of Numerical Features')

# 13. Title length vs. description length
plt.subplot(3, 3, 4)
plt.scatter(df_processed['title_length'], df_processed['description_length'], alpha=0.3)
plt.xlabel('Title Length')
plt.ylabel('Description Length')
plt.title('Title Length vs. Description Length')

# 14. Platform distribution
plt.subplot(3, 3, 5)
platform_counts = df_processed['platform'].value_counts()
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Content by Platform')

# 15. Day of week added distribution
plt.subplot(3, 3, 6)
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
               4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df_processed['day_name'] = df_processed['day_of_week_added'].map(day_mapping)
day_counts = df_processed['day_name'].value_counts()
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.title('Content Added by Day of Week')
plt.xticks(rotation=45)

# 16. Month added distribution
plt.subplot(3, 3, 7)
month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df_processed['month_name'] = df_processed['month_added'].map(month_mapping)
month_counts = df_processed['month_name'].value_counts()
month_counts = month_counts.reindex(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
sns.barplot(x=month_counts.index, y=month_counts.values)
plt.title('Content Added by Month')
plt.xticks(rotation=45)

# 17. Distribution of movies and shows by release year
plt.subplot(3, 3, 8)
sns.histplot(data=df_processed, x='release_year', hue='type', multiple='stack', bins=30)
plt.title('Movies and Shows by Release Year')
plt.xticks(rotation=45)

plt.savefig('Visualizations/Feature_Engineering/feature_visualizations_2.png', bbox_inches='tight')

# Create a function to select the most relevant features for recommendation system
def select_recommendation_features(df):
    content_features = [
        'platform', 'type', 'release_year', 'content_age', 'duration_numeric', 
        'duration_unit', 'min_age', 'age_category', 'decade', 'genre_count' ]
    
    content_features += [f'genre_{g}' for g in top_genres]
    content_features += ['title_length', 'description_length'] 
    content_features += [f'tfidf_pca_{i}' for i in range(5)]
    content_features += ['director_count', 'cast_count', 'has_director', 'has_cast']
    content_features += ['recency_score', 'popularity_proxy', 'movie_duration', 'show_duration']
    selected_features = df[content_features].copy()
    selected_numeric = selected_features.select_dtypes(include=['number'])
    
    plt.figure(figsize=(20, 16))
    corr_matrix = selected_numeric.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                linewidths=0.5, vmax=1.0, vmin=-1.0)
    plt.title('Correlation Matrix of Selected Features for Recommendation')
    plt.tight_layout()
    plt.savefig('Visualizations/Feature_Engineering/recommendation_features_correlation.png')
    
    return selected_features, content_features

selected_features, feature_list = select_recommendation_features(df_processed)
selected_features.to_csv('recommendation_features.csv', index=False)

print("Engineered Features for Recommendation System:")
for i, feature in enumerate(feature_list, 1):
    print(f"{i}. {feature}")