import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import re
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ContentRecommendationModel')


class ContentStreamingRecommendationModel:
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.features_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.label_encoder = None
        self.content_clusters = None
        self.similarity_model = None
        self.genre_similarity_matrix = None
        self.all_genres = set()
        self.title_embeddings = {}

        self.random_state = 42
        self.n_clusters = 40  
        self.n_neighbors = 200 
        
        self.content_features = [
            'content_age', 'release_decade', 'genre_count', 'director_count',
            'cast_size', 'is_movie', 'country_count', 'is_us',
            'duration_normalized', 'description_length', 'title_length'
        ]
        
        self.clustering_features = [
            'genre_count', 'content_age', 'is_movie', 'duration_normalized',
            'description_length', 'release_decade', 'is_us', 'director_count', 'cast_size'
        ]
        
        self.recommendation_weights = {
            'content_weight': 0.35,    
            'genre_weight': 0.40,     
            'cluster_weight': 0.05,    
            'thematic_weight': 0.20   
        }
        

        self.rating_compatibility_rules = {
            ('Adult', 'Adult'): 1.0,
            ('Adult', 'Teen'): 0.8,
            ('Adult', 'Family'): 0.0,
            ('Adult', 'Kids'): 0.0,
            ('Teen', 'Teen'): 1.0,
            ('Teen', 'Adult'): 0.6,
            ('Teen', 'Family'): 0.7,
            ('Teen', 'Kids'): 0.4,
            ('Family', 'Family'): 1.0,
            ('Family', 'Kids'): 0.9,
            ('Family', 'Teen'): 0.7,
            ('Family', 'Adult'): 0.0,
            ('Kids', 'Kids'): 1.0,
            ('Kids', 'Family'): 0.8,
            ('Kids', 'Teen'): 0.0,
            ('Kids', 'Adult'): 0.0,
            ('Unknown', 'Unknown'): 0.7,
            ('Unknown', 'Adult'): 0.5,
            ('Unknown', 'Teen'): 0.6,
            ('Unknown', 'Family'): 0.6,
            ('Unknown', 'Kids'): 0.5
        }
        
        self.platform_diversity_weights = {
            'Netflix': 1.1,       
            'Amazon Prime': 0.9,  
            'Hulu': 1.2           
        }
    
    def save_model(self, filepath: str) -> None:
        model_data = {
            'content_clusters': self.content_clusters,
            'similarity_model': self.similarity_model,
            'genre_similarity_matrix': self.genre_similarity_matrix,
            'all_genres': self.all_genres,
            'content_features': self.content_features,
            'clustering_features': self.clustering_features,
            'recommendation_weights': self.recommendation_weights,
            'n_clusters': self.n_clusters,
            'n_neighbors': self.n_neighbors,
            'random_state': self.random_state,
            'title_embeddings': self.title_embeddings,
            'rating_compatibility_rules': self.rating_compatibility_rules
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.content_clusters = model_data['content_clusters']
        self.similarity_model = model_data['similarity_model']
        self.genre_similarity_matrix = model_data.get('genre_similarity_matrix')
        self.all_genres = model_data.get('all_genres', set())
        self.content_features = model_data['content_features']
        self.clustering_features = model_data['clustering_features']
        self.recommendation_weights = model_data.get('recommendation_weights', {
            'content_weight': 0.35,    
            'genre_weight': 0.40,     
            'cluster_weight': 0.05,    
            'thematic_weight': 0.20 
        })
        self.n_clusters = model_data['n_clusters']
        self.n_neighbors = model_data['n_neighbors']
        self.random_state = model_data['random_state']
        self.title_embeddings = model_data.get('title_embeddings', {})
        self.rating_compatibility_rules = model_data.get('rating_compatibility_rules', {})
        
        logger.info(f"Model loaded from {filepath}")
    
    def load_data(self, filepath: str) -> int:
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.data)} records from {filepath}")
            self.preprocess_data()
            return len(self.data)
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def preprocess_data(self) -> None:
        logger.info("Preprocessing data for ML modeling")
        
        df = self.data.copy()
        
        for feature in self.content_features:
            if feature in df.columns:
                if df[feature].dtype in [np.float64, np.int64]:
                    df[feature] = df[feature].fillna(df[feature].median())
                else:
                    df[feature] = df[feature].fillna('')
        
        self.label_encoder = LabelEncoder()
        if 'rating_category' in df.columns:
            df['rating_category'] = df['rating_category'].fillna('Unknown')
            df['rating_category_encoded'] = self.label_encoder.fit_transform(df['rating_category'])
        
        df['processed_genres'] = df['genre_list'].apply(self._parse_genres)
        
        all_genres = set()
        for genres in df['processed_genres']:
            all_genres.update(genres)
        self.all_genres = all_genres
        logger.info(f"Found {len(self.all_genres)} unique genres")
        
        self._create_data_splits(df)
        self._build_genre_similarity_matrix()
        logger.info("Extracting text features")
        self._extract_text_features(df)
        self.features_df = df.copy()
        
        self.processed_data = []
        
        for i, row in df.iterrows():
            content_attrs = {}
            for feature in self.content_features:
                if feature in row and not pd.isna(row[feature]):
                    content_attrs[feature] = row[feature]
                else:
                    content_attrs[feature] = 0
            
            rating_category = 'Unknown'
            if 'rating_category' in row and not pd.isna(row['rating_category']):
                rating_category = row['rating_category']
            elif 'rating' in row and not pd.isna(row['rating']):
                rating = str(row['rating']).upper()
                if rating in ['TV-Y', 'TV-Y7', 'G', 'TV-G']:
                    rating_category = 'Kids'
                elif rating in ['PG', 'TV-PG']:
                    rating_category = 'Family'
                elif rating in ['PG-13', 'TV-14']:
                    rating_category = 'Teen'
                elif rating in ['R', 'TV-MA', 'NC-17']:
                    rating_category = 'Adult'
            
            self.processed_data.append({
                'id': row['show_id'],
                'title': row['title'],
                'platform': row['platform'],
                'type': row['type'],
                'content_features': content_attrs,
                'rating': row.get('rating', 'Unknown'),
                'rating_category': rating_category,
                'genres': row['processed_genres'],
                'description': row.get('description', ''),
                'cluster': -1,  
                'split': row.get('split', 'train')  
            })
        
        self._train_models()
        
        self._save_data_splits()
    
    def _create_data_splits(self, df):
        logger.info("Creating train/val/test splits with 8:1:1 ratio")
        
        df['stratify_group'] = df['platform'] + "_" + df['type']

        temp_df = df[['show_id', 'stratify_group']].copy()

        train_val_idx, test_idx = train_test_split(
            temp_df.index, 
            test_size=0.1, 
            random_state=self.random_state, 
            stratify=temp_df['stratify_group']
        )

        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=1/9, 
            random_state=self.random_state, 
            stratify=temp_df.loc[train_val_idx, 'stratify_group']
        )

        df['split'] = 'train'  # Default
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self.train_data = df[df['split'] == 'train'].copy()
        self.val_data = df[df['split'] == 'val'].copy()
        self.test_data = df[df['split'] == 'test'].copy()

        total = len(df)
        train_pct = len(self.train_data) / total * 100
        val_pct = len(self.val_data) / total * 100
        test_pct = len(self.test_data) / total * 100
        
        logger.info(f"Split created - Train: {len(self.train_data)} ({train_pct:.1f}%), "
                   f"Val: {len(self.val_data)} ({val_pct:.1f}%), "
                   f"Test: {len(self.test_data)} ({test_pct:.1f}%)")
        
        for split_name, split_df in [('Train', self.train_data), ('Val', self.val_data), ('Test', self.test_data)]:
            platform_counts = split_df['platform'].value_counts(normalize=True) * 100
            type_counts = split_df['type'].value_counts(normalize=True) * 100
            
            logger.info(f"{split_name} set platform distribution: " + 
                       ", ".join(f"{p}: {v:.1f}%" for p, v in platform_counts.items()))
            
            logger.info(f"{split_name} set content type distribution: " + 
                       ", ".join(f"{t}: {v:.1f}%" for t, v in type_counts.items()))
    
    def _save_data_splits(self):
        if not os.path.exists('Data'):
            os.makedirs('Data')
        
        self.train_data.to_csv('Data/train_data.csv', index=False)
        self.val_data.to_csv('Data/val_data.csv', index=False)
        self.test_data.to_csv('Data/test_data.csv', index=False)
        
        logger.info(f"Saved data splits to Data/ directory")
    
    def _extract_text_features(self, df: pd.DataFrame) -> None:
        for i, row in df.iterrows():
            if pd.isna(row['title']):
                continue
                
            title = row['title'].lower()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
            title_words = re.findall(r'\b\w+\b', title)
            title_words = [word for word in title_words if word not in common_words and len(word) > 2]
            self.title_embeddings[row['show_id']] = title_words
    
    def _parse_genres(self, genre_list: Union[str, List[str]]) -> List[str]:
        if pd.isna(genre_list):
            return []
            
        genres = []
        
        try:
            if isinstance(genre_list, str):
                genre_list = genre_list.replace("'", '"')
                try:
                    parsed = ast.literal_eval(genre_list)
                    genres = parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, SyntaxError):
                    genres = [g.strip() for g in genre_list.strip('[]').split(',')]
            elif isinstance(genre_list, list):
                genres = genre_list
        except Exception:
            if isinstance(genre_list, str):
                genres = [g.strip() for g in genre_list.strip('[]').split(',')]
            elif isinstance(genre_list, list):
                genres = genre_list
        
        cleaned_genres = []
        genre_mapping = {
            'comedies': 'comedy', 'comedy movies': 'comedy', 'dramas': 'drama',
            'drama movies': 'drama', 'documentaries': 'documentary',
            'documentary films': 'documentary', 'sci-fi': 'science fiction',
            'scifi': 'science fiction', 'sci fi': 'science fiction',
            'science-fiction': 'science fiction', 'action & adventure': 'action',
            'action and adventure': 'action', 'children': "children's",
            'kids': "children's", 'stand-up': 'stand-up comedy',
            'stand up comedy': 'stand-up comedy', 'music & musicals': 'musical',
            'thrillers': 'thriller', 'thriller movies': 'thriller',
            'romantic comedies': 'romantic comedy', 'horror movies': 'horror',
            'crime films': 'crime', 'crime': 'crime thriller',
            'romantic dramas': 'romance', 'romance movies': 'romance',
            'anime': 'animation', 'animated': 'animation', 'cartoon': 'animation',
            'sitcoms': 'sitcom', 'reality tv': 'reality', 'food & wine': 'food',
            'historical': 'history'
        }
        
        for genre in genres:
            if not genre or not isinstance(genre, str):
                continue
                
            g = genre.strip().strip('"\'').lower()
            if not g:
                continue
                
            matched = False
            for k, v in genre_mapping.items():
                if k == g or k in g:
                    g = v
                    matched = True
                    break
            
            if not matched:
                if 'comedy' in g:
                    g = 'comedy'
                elif 'drama' in g:
                    g = 'drama'
                elif 'action' in g:
                    g = 'action'
                elif 'documentary' in g:
                    g = 'documentary'
            
            cleaned_genres.append(g)
        
        return list(set(cleaned_genres))
    
    def _build_genre_similarity_matrix(self) -> None:
        genres_list = list(self.all_genres)
        n_genres = len(genres_list)
        similarity_matrix = np.zeros((n_genres, n_genres))

        genre_relationships = {
            'comedy': ['romantic comedy', 'sitcom', 'stand-up comedy', 'satire'],
            'drama': ['crime drama', 'period drama', 'thriller', 'romance'],
            'action': ['adventure', 'thriller', 'science fiction', 'crime'],
            'horror': ['thriller', 'suspense', 'supernatural', 'slasher'],
            'documentary': ['biography', 'history', 'nature', 'science'],
            'romance': ['romantic comedy', 'drama', 'period drama'],
            'thriller': ['crime', 'suspense', 'action', 'mystery'],
            'animation': ["children's", 'anime', 'family'],
            "children's": ['family', 'animation', 'educational'],
            'science fiction': ['fantasy', 'action', 'adventure', 'dystopian'],
            'fantasy': ['adventure', 'science fiction', 'supernatural'],
            'crime thriller': ['thriller', 'drama', 'mystery', 'detective'],
            'mystery': ['thriller', 'crime', 'suspense', 'detective'],
            'history': ['period drama', 'war', 'biography'],
            'musical': ['dance', 'concert', 'romance'],
            'sports': ['documentary', 'drama', 'competition'],
            'war': ['drama', 'action', 'historical'],
            'western': ['action', 'drama', 'adventure'],
            'reality': ['competition', 'documentary', 'talk show'],
            'stand-up comedy': ['comedy', 'talk show'],
            'sitcom': ['comedy', 'family'],
            'superhero': ['action', 'science fiction', 'fantasy'],
            'anime': ['animation', 'fantasy', 'action'],
            'food': ['reality', 'documentary', 'cooking'],
            'competition': ['reality', 'game show', 'sports'],
            'talk show': ['comedy', 'reality', 'interview'],
            'family': ["children's", 'comedy', 'drama']
        }
  
        for i, genre1 in enumerate(genres_list):
            for j, genre2 in enumerate(genres_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
    
                tokens1 = set(genre1.split())
                tokens2 = set(genre2.split())
                jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2)) if tokens1.union(tokens2) else 0
           
                related = False
                relatedness = 0.0
                
                if genre1 in genre_relationships and genre2 in genre_relationships[genre1]:
                    related = True
                    relatedness = 0.85
                elif genre2 in genre_relationships and genre1 in genre_relationships[genre2]:
                    related = True
                    relatedness = 0.85
          
                if not related and genre1 in genre_relationships and genre2 in genre_relationships:
                    common_related = set(genre_relationships[genre1]).intersection(set(genre_relationships[genre2]))
                    if common_related:
                        related = True
                        relatedness = 0.7
                        if len(common_related) >= 3:
                            relatedness = 0.8
                
                similarity = max(jaccard, relatedness if related else 0)
                similarity_matrix[i, j] = similarity
        
        self.genre_similarity_matrix = {
            'matrix': similarity_matrix,
            'genres': genres_list
        }
        
        logger.info("Built enhanced genre similarity matrix")
    
    def _train_models(self) -> None:
        logger.info("Training machine learning models")
        self._train_content_clustering()
        self._train_similarity_model()
        logger.info("Model training complete")
    
    def _train_content_clustering(self) -> None:
        logger.info(f"Training content clustering with {self.n_clusters} clusters")
        
        df = self.train_data.copy()
        
        cluster_features = [f for f in self.clustering_features if f in df.columns]
        X = df[cluster_features].copy().fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_features)
        
        pca = None
        if X_scaled_df.shape[1] > 8:
            pca = PCA(n_components=8, random_state=self.random_state)
            X_reduced = pca.fit_transform(X_scaled_df)
            explained_variance = sum(pca.explained_variance_ratio_)
            logger.info(f"PCA explained variance: {explained_variance:.4f}")
            X_reduced_df = pd.DataFrame(X_reduced, columns=[f"pc_{i}" for i in range(8)])
            X_for_clustering = X_reduced_df
        else:
            X_for_clustering = X_scaled_df
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto',
            max_iter=500,
            tol=1e-5
        )
        
        clusters = kmeans.fit_predict(X_for_clustering)
        self.features_df['cluster'] = -1  
        
        train_indices = df.index
        for i, idx in enumerate(train_indices):
            self.features_df.loc[idx, 'cluster'] = int(clusters[i])

        for split_df in [self.val_data, self.test_data]:
            if len(split_df) > 0:
                X_split = split_df[cluster_features].copy().fillna(0)
                X_split_scaled = scaler.transform(X_split)
                
                if pca is not None:
                    X_split_reduced = pca.transform(X_split_scaled)
                    X_split_for_clustering = pd.DataFrame(X_split_reduced, columns=[f"pc_{i}" for i in range(8)])
                else:
                    X_split_for_clustering = pd.DataFrame(X_split_scaled, columns=cluster_features)
                
                split_clusters = kmeans.predict(X_split_for_clustering)
                
                for i, idx in enumerate(split_df.index):
                    self.features_df.loc[idx, 'cluster'] = int(split_clusters[i])
        
        for i, item in enumerate(self.processed_data):
            idx = self.features_df[self.features_df['show_id'] == item['id']].index
            if len(idx) > 0:
                item['cluster'] = int(self.features_df.loc[idx[0], 'cluster'])
        
        self.content_clusters = {
            'model': kmeans,
            'scaler': scaler,
            'pca': pca,
            'features': cluster_features,
            'cluster_counts': np.bincount(clusters).tolist()
        }
        
        logger.info(f"Cluster distribution: {np.bincount(clusters)}")
        self._create_cluster_profiles(self.features_df)
    
    def _create_cluster_profiles(self, df: pd.DataFrame) -> None:
        cluster_profiles = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            if len(cluster_data) == 0:
                continue
                
            all_genres = []
            for genres in cluster_data['processed_genres']:
                all_genres.extend(genres)
            genre_counter = Counter(all_genres)
            top_genres = genre_counter.most_common(5)
            
            movie_count = cluster_data['is_movie'].sum()
            tv_count = len(cluster_data) - movie_count
            
            platform_counts = cluster_data['platform'].value_counts().to_dict()
            avg_age = cluster_data['content_age'].mean()
            
            cluster_profiles[cluster_id] = {
                'size': len(cluster_data),
                'top_genres': top_genres,
                'movie_ratio': movie_count / len(cluster_data) if len(cluster_data) > 0 else 0,
                'tv_ratio': tv_count / len(cluster_data) if len(cluster_data) > 0 else 0,
                'platforms': platform_counts,
                'avg_content_age': avg_age
            }
        
        self.content_clusters['profiles'] = cluster_profiles
        
        for cluster_id in list(cluster_profiles.keys())[:3]:
            profile = cluster_profiles[cluster_id]
            genres_str = ", ".join([f"{g[0]} ({g[1]})" for g in profile['top_genres']])
            logger.info(f"Cluster {cluster_id}: {profile['size']} items, "
                       f"Genres: {genres_str}, "
                       f"Movie:TV = {profile['movie_ratio']:.2f}:{profile['tv_ratio']:.2f}")
    
    def _train_similarity_model(self) -> None:

        logger.info("Training nearest neighbors model for content similarity")
        
        df = self.train_data.copy()
        
        available_features = [f for f in self.content_features if f in df.columns]
        
        if len(available_features) < len(self.content_features):
            missing_features = [f for f in self.content_features if f not in df.columns]
            logger.warning(f"Missing features: {missing_features}. Continuing with available features only.")

            for feature in missing_features:
                df[feature] = 0
        
        X = df[self.content_features].copy().fillna(0)
        feature_names = X.columns.tolist()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, X.shape[0]),
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        
        nn_model.fit(X_scaled)
        
        self.similarity_model = {
            'model': nn_model,
            'scaler': scaler,
            'features': feature_names,
            'indices': df.index.tolist()
        }
    
    def calculate_rating_compatibility(self, source_rating: str, target_rating: str) -> float:
        return self.rating_compatibility_rules.get(
            (source_rating, target_rating),
            self.rating_compatibility_rules.get(
                ('Unknown', 'Unknown'), 0.5
            )
        )
    
    def calculate_content_similarity(self, item1: Dict, item2: Dict) -> float:
        feature_weights = {
            'is_movie': 5.0,
            'genre_count': 1.5,
            'content_age': 3.0,
            'duration_normalized': 2.0,
            'director_count': 1.2,
            'cast_size': 1.5,
            'is_us': 0.5,
            'country_count': 0.8,
            'release_decade': 3.0,
            'description_length': 1.0,
            'title_length': 0.3,
            'is_exclusive': 0.8
        }
        
        weighted_sum = 0
        weight_sum = 0
        
        for feature in self.content_features:
            weight = feature_weights.get(feature, 1)
            if (feature in item1['content_features'] and 
                feature in item2['content_features']):
                
                val1 = item1['content_features'][feature]
                val2 = item2['content_features'][feature]
                
                if feature == 'is_movie':
                    type_match = 1 if val1 == val2 else 0
                    weighted_sum += weight * (1 - type_match) * (1 - type_match)
                else:
                    feature_range = 1.0
                    if feature == 'content_age':
                        feature_range = 50.0
                    elif feature == 'duration_normalized':
                        feature_range = 2.0
                    elif feature in ['cast_size', 'director_count']:
                        feature_range = 10.0
                    
                    norm_diff = abs(val1 - val2) / feature_range
                    weighted_sum += weight * norm_diff * norm_diff
                
                weight_sum += weight
        
        if weight_sum == 0:
            return 0
            
        distance = np.sqrt(weighted_sum / weight_sum)
        similarity = 1 / (1 + (distance * 1.5))
        
        if 'rating_category' in item1 and 'rating_category' in item2:
            rating_compat = self.calculate_rating_compatibility(
                item1['rating_category'],
                item2['rating_category']
            )
            similarity *= rating_compat

        title_sim = self._calculate_title_similarity(item1['id'], item2['id'])
        similarity = similarity * (1.0 + 0.2 * title_sim)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_title_similarity(self, id1: str, id2: str) -> float:
        keywords1 = self.title_embeddings.get(id1, [])
        keywords2 = self.title_embeddings.get(id2, [])
        
        if not keywords1 or not keywords2:
            return 0.0
        
        set1 = set(keywords1)
        set2 = set(keywords2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_genre_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        if not genres1 or not genres2:
            return 0.0
            
        def clean_genres(genres_list):
            if not genres_list:
                return []
            return [g.strip().lower() for g in genres_list if isinstance(g, str) and g.strip()]
            
        genres1_clean = clean_genres(genres1)
        genres2_clean = clean_genres(genres2)
        
        if not genres1_clean or not genres2_clean:
            return 0.0
        
        set1 = set(genres1_clean)
        set2 = set(genres2_clean)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        if jaccard > 0:
            direct_match_score = min(1.0, jaccard * 1.5)
            if jaccard > 0.5:
                return min(1.0, direct_match_score * 1.2)
            return direct_match_score
        
        if self.genre_similarity_matrix:
            matrix = self.genre_similarity_matrix['matrix']
            all_genres = self.genre_similarity_matrix['genres']
            
            indices1 = [i for i, g in enumerate(all_genres) if g in genres1_clean]
            indices2 = [i for i, g in enumerate(all_genres) if g in genres2_clean]
            
            if not indices1 or not indices2:
                return 0.0
                
            primary_genres = {'comedy', 'drama', 'action', 'thriller', 'horror', 
                            'science fiction', 'documentary', 'animation', 'adventure'}
            
            best_matches = []
            for i in indices1:
                genre1 = all_genres[i]
                max_sim = 0
                for j in indices2:
                    genre2 = all_genres[j]
                    sim = matrix[i, j]
                    max_sim = max(max_sim, sim)
                
                weight = 1.5 if genre1 in primary_genres else 1.0
                best_matches.append((max_sim, weight))
            
            total_weight = sum(w for _, w in best_matches)
            if total_weight == 0:
                return 0.0
                
            weighted_sim = sum(sim * w for sim, w in best_matches) / total_weight
            return min(1.0, weighted_sim * 1.2)
        
        return 0.0
    
    def calculate_cluster_similarity(self, cluster1: int, cluster2: int) -> float:
        if cluster1 == -1 or cluster2 == -1:
            return 0.5
            
        if cluster1 == cluster2:
            return 1.0
        
        if hasattr(self, 'content_clusters') and 'profiles' in self.content_clusters:
            profiles = self.content_clusters['profiles']
            
            if cluster1 in profiles and cluster2 in profiles:
                profile1 = profiles[cluster1]
                profile2 = profiles[cluster2]
                
                genres1 = set(g[0] for g in profile1['top_genres'])
                genres2 = set(g[0] for g in profile2['top_genres'])
                genre_overlap = len(genres1.intersection(genres2))
                genre_sim = genre_overlap / max(len(genres1), len(genres2)) if max(len(genres1), len(genres2)) > 0 else 0
                
                type_sim = 1 - abs(profile1['movie_ratio'] - profile2['movie_ratio'])
                age_diff = abs(profile1['avg_content_age'] - profile2['avg_content_age'])
                age_sim = max(0, 1 - (age_diff / 20))
                
                cluster_sim = 0.5 * genre_sim + 0.3 * type_sim + 0.2 * age_sim
                return max(0.3, cluster_sim)
        
        return 0.3
    
    def calculate_thematic_similarity(self, item1: Dict, item2: Dict) -> float:
        if not item1 or not item2:
            return 0.5
        
        theme_categories = {
            'comedy': ['funny', 'humor', 'laugh', 'sitcom', 'hilarious'],
            'drama': ['dramatic', 'emotional', 'serious', 'intense'],
            'action': ['adventure', 'fight', 'mission', 'hero', 'battle'],
            'thriller': ['suspense', 'mystery', 'tension', 'crime'],
            'horror': ['scary', 'terror', 'supernatural', 'monster'],
            'science fiction': ['future', 'space', 'alien', 'robot'],
            'fantasy': ['magic', 'kingdom', 'creature', 'myth'],
            'documentary': ['real', 'history', 'fact', 'educational'],
            'romance': ['love', 'relationship', 'couple', 'romantic'],
            'family': ['kids', 'children', 'parents', 'family-friendly'],
            'crime': ['police', 'criminal', 'investigation', 'murder'],
            'sports': ['competition', 'athlete', 'game', 'team'],
            'historical': ['history', 'period', 'era', 'war'],
            'musical': ['music', 'song', 'dance', 'singing'],
            'animation': ['cartoon', 'animated', 'anime', 'drawing'],
            'reality': ['competition', 'contestant', 'real life', 'challenge']
        }
        
        def extract_themes_from_genres(genres):
            themes = set()
            for genre in genres:
                genre_lower = genre.lower()
                for category, keywords in theme_categories.items():
                    if any(keyword in genre_lower for keyword in [category] + keywords):
                        themes.add(category)
            return themes
        
        def extract_type_themes(content_type, description=""):
            type_themes = set()
            if not description:
                return type_themes
                
            description = description.lower()
            
            if content_type == "TV Show":
                tv_themes = {
                    'sitcom': ['comedy', 'ensemble cast', 'episode'],
                    'drama series': ['emotional', 'serious', 'character development'],
                    'procedural': ['case', 'solve', 'crime', 'detective'],
                    'teen drama': ['high school', 'teenager', 'coming of age']
                }
                for theme, keywords in tv_themes.items():
                    if any(keyword in description for keyword in keywords):
                        type_themes.add(theme)
            elif content_type == "Movie":
                movie_themes = {
                    'blockbuster': ['action', 'adventure', 'spectacle'],
                    'indie': ['low budget', 'character study', 'artistic'],
                    'thriller': ['suspense', 'twist', 'mystery'],
                    'romance': ['love story', 'romantic', 'relationship']
                }
                for theme, keywords in movie_themes.items():
                    if any(keyword in description for keyword in keywords):
                        type_themes.add(theme)
                        
            return type_themes
        
        item1_themes = extract_themes_from_genres(item1.get('genres', []))
        item2_themes = extract_themes_from_genres(item2.get('genres', []))
        
        item1_type = item1.get('type', '')
        item2_type = item2.get('type', '')
        item1_desc = item1.get('description', '')
        item2_desc = item2.get('description', '')
        
        item1_themes.update(extract_type_themes(item1_type, item1_desc))
        item2_themes.update(extract_type_themes(item2_type, item2_desc))
        
        if not item1_themes or not item2_themes:
            return 0.5
            
        theme_similarity = len(item1_themes.intersection(item2_themes)) / len(item1_themes.union(item2_themes))
        
        desc_similarity = 0.

        # Calculate description similarity if both items have descriptions
        if item1_desc and item2_desc:
            desc1 = item1_desc.lower()
            desc2 = item2_desc.lower()
            
            keyword_matches = 0
            total_keywords = 0
            
            for category, keywords in theme_categories.items():
                keywords_to_check = [category] + keywords
                for keyword in keywords_to_check:
                    keyword_in_desc1 = keyword in desc1
                    keyword_in_desc2 = keyword in desc2
                    
                    if keyword_in_desc1 and keyword_in_desc2:
                        keyword_matches += 1
                    if keyword_in_desc1 or keyword_in_desc2:
                        total_keywords += 1
            
            if total_keywords > 0:
                desc_similarity = keyword_matches / total_keywords

        type_boost = 1.2 if item1_type == item2_type else 1.0

        combined_similarity = (0.7 * theme_similarity + 0.3 * desc_similarity) * type_boost
        
        return min(1.0, combined_similarity)

    def find_similar_items(self, content_id: str, max_items: int = 50, diversity_factor: float = 0.5, recursion_depth: int = 0) -> List[Dict]:
        if recursion_depth >= 3:
            logger.warning(f"Maximum recursion depth reached for content_id {content_id}")
            return [] 
        
        if not self.similarity_model:
            raise Exception("Similarity model not trained")
        
        source_item = None
        for i, item in enumerate(self.processed_data):
            if item['id'] == content_id:
                source_item = item
                break
        
        if not source_item:
            raise Exception(f"Content with ID {content_id} not found")

        feature_values = []
        for feature in self.similarity_model['features']:
            value = source_item['content_features'].get(feature, 0)
            feature_values.append(value)

        features_array = np.array([feature_values])
        scaler = self.similarity_model['scaler']
        features_scaled = scaler.transform(features_array)

        nn_model = self.similarity_model['model']
        distances, indices = nn_model.kneighbors(features_scaled, n_neighbors=min(200, len(self.processed_data)))

        training_indices = self.similarity_model['indices']
        actual_indices = [training_indices[idx] for idx in indices[0]]

        source_platform = source_item['platform']
        source_genres = source_item['genres']
        source_cluster = source_item['cluster']
        source_rating = source_item.get('rating_category', 'Unknown')
        source_type = source_item['type']
        
        similar_items = []
        platform_counts = {'Netflix': 0, 'Amazon Prime': 0, 'Hulu': 0}
        type_counts = {'Movie': 0, 'TV Show': 0}
        skipped_count = 0
        
        for i, idx in enumerate(actual_indices):
            item_to_check = None
            for item in self.processed_data:
                if self.features_df.loc[idx, 'show_id'] == item['id']:
                    item_to_check = item
                    break
            
            if not item_to_check or item_to_check['id'] == source_item['id']:
                continue
                
            distance = distances[0][i]
            nn_similarity = np.exp(-distance)
            
            item_rating = item_to_check.get('rating_category', 'Unknown')
            rating_compat = self.calculate_rating_compatibility(source_rating, item_rating)
            rating_threshold = max(0.1, 0.3 - (recursion_depth * 0.1))
            if rating_compat <= rating_threshold:
                skipped_count += 1
                continue
            
            content_sim = self.calculate_content_similarity(source_item, item_to_check)
            genre_sim = self.calculate_genre_similarity(source_genres, item_to_check['genres'])
            cluster_sim = self.calculate_cluster_similarity(source_cluster, item_to_check['cluster'])
            thematic_sim = self.calculate_thematic_similarity(source_item, item_to_check)
            
            content_threshold = max(0.05, 0.1 - (recursion_depth * 0.03))
            genre_threshold = max(0.05, 0.15 - (recursion_depth * 0.05))
            if content_sim < content_threshold and genre_sim < genre_threshold:
                skipped_count += 1
                continue
            
            weights = self.recommendation_weights
            hybrid_score = (
                weights['content_weight'] * content_sim +
                weights['genre_weight'] * genre_sim +
                weights['cluster_weight'] * cluster_sim +
                weights['thematic_weight'] * thematic_sim
            ) / sum(weights.values())
            
            hybrid_score *= rating_compat
            
            platform_bonus = 0.2 if item_to_check['platform'] != source_platform else 0
            type_bonus = 0.15 if item_to_check['type'] == source_type else -0.05
            
            adjusted_score = (hybrid_score * (1 - diversity_factor) + 
                            (platform_bonus + type_bonus) * diversity_factor)
            
            # Platform balancing - relax constraints when recursing
            max_platform_count = max_items // 3 + recursion_depth
            if platform_counts[item_to_check['platform']] > max_platform_count:
                adjusted_score *= 0.9
            
            # Type balancing - relax constraints when recursing
            max_type_count = max_items // 2 + recursion_depth
            if type_counts[item_to_check['type']] > max_type_count:
                adjusted_score *= 0.9
            
            similar_items.append({
                'id': item_to_check['id'],
                'title': item_to_check['title'],
                'platform': item_to_check['platform'],
                'type': item_to_check['type'],
                'genres': item_to_check['genres'],
                'cluster': item_to_check['cluster'],
                'rating': item_to_check.get('rating', 'Unknown'),
                'rating_category': item_rating,
                'distance': distance,
                'nn_similarity': nn_similarity,
                'content_similarity': content_sim,
                'genre_similarity': genre_sim,
                'cluster_similarity': cluster_sim,
                'thematic_similarity': thematic_sim,
                'hybrid_score': hybrid_score,
                'adjusted_score': adjusted_score,
                'rating_compatibility': rating_compat,
                'split': item_to_check.get('split', 'unknown')
            })
            
            platform_counts[item_to_check['platform']] += 1
            type_counts[item_to_check['type']] += 1

        min_results = max(max_items // 2, 10)
        if len(similar_items) < min_results and skipped_count > 0 and recursion_depth < 2:
            logger.info(f"Relaxing filters (found {len(similar_items)} items, skipped {skipped_count})")
            new_diversity_factor = diversity_factor * 0.5
            additional_items = self.find_similar_items(
                content_id, 
                max_items=max_items, 
                diversity_factor=new_diversity_factor,
                recursion_depth=recursion_depth + 1
            )
            
            existing_ids = {item['id'] for item in similar_items}
            for item in additional_items:
                if item['id'] not in existing_ids:
                    similar_items.append(item)
                    existing_ids.add(item['id'])
        
        similar_items.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return similar_items[:max_items]
    
    def get_recommendations(self, content_id: str, options: Optional[Dict] = None) -> Dict:
            if options is None:
                options = {}
            
            max_recommendations = options.get('max_recommendations', 5)
            cross_platform_only = options.get('cross_platform_only', False)
            include_metrics = options.get('include_metrics', True)
            platform_diversity = options.get('platform_diversity', True)
            same_type_boost = options.get('same_type_boost', True)
            strict_type_matching = options.get('strict_type_matching', True)
            
            source_content = None
            for item in self.processed_data:
                if item['id'] == content_id:
                    source_content = item
                    break
            
            if not source_content:
                return {'error': 'Content not found'}
            
            similar_items = self.find_similar_items(
                content_id,
                max_items=max(100, max_recommendations * 8),
                diversity_factor=0.3 if platform_diversity else 0.0
            )
            
            recommendations = []
            platform_counts = {'Netflix': 0, 'Amazon Prime': 0, 'Hulu': 0}
            type_counts = {'Movie': 0, 'TV Show': 0}
            
            total_weight = sum(self.platform_diversity_weights.values())
            max_per_platform = {}
            for platform, weight in self.platform_diversity_weights.items():
                normalized_weight = weight / total_weight
                max_per_platform[platform] = max(2, int(max_recommendations * normalized_weight * 2))
            
            if strict_type_matching:
                strict_type_items = [item for item in similar_items if item['type'] == source_content['type']]
                
                if len(strict_type_items) < max_recommendations * 2:
                    high_sim_items = [
                        item for item in similar_items 
                        if (item['type'] != source_content['type'] and
                            item.get('genre_similarity', 0) > 0.8 and
                            item.get('content_similarity', 0) > 0.6)
                    ]
                    
                    max_diff_type = max(1, int(max_recommendations * 0.25))
                    strict_type_items.extend(high_sim_items[:max_diff_type])
                    
                similar_items = strict_type_items
            
            min_per_platform = max(1, max_recommendations // 6)
            for platform in platform_counts.keys():
                platform_items = [item for item in similar_items 
                                if item['platform'] == platform 
                                and (not cross_platform_only or item['platform'] != source_content['platform'])]
                
                platform_items.sort(key=lambda x: x['hybrid_score'] * self.platform_diversity_weights[platform], reverse=True)
                
                for item in platform_items[:min_per_platform]:
                    if platform_counts[item['platform']] < min_per_platform and item not in recommendations:
                        type_boost = 1.25 if (same_type_boost and 
                                            item['type'] == source_content['type']) else 1.0
                        
                        final_score = item['hybrid_score'] * type_boost * self.platform_diversity_weights[item['platform']]
                        
                        result = {
                            'id': item['id'],
                            'title': item['title'],
                            'platform': item['platform'],
                            'type': item['type'],
                            'score': final_score,
                            'genres': item['genres'],
                            'cluster': item['cluster'],
                            'rating_category': item.get('rating_category', 'Unknown'),
                            'split': item.get('split', 'unknown')
                        }
                        
                        if include_metrics:
                            result['metrics'] = {
                                'content_similarity': item['content_similarity'],
                                'genre_similarity': item['genre_similarity'],
                                'cluster_similarity': item['cluster_similarity'],
                                'thematic_similarity': item['thematic_similarity'],
                                'rating_compatibility': item['rating_compatibility']
                            }
                        
                        recommendations.append(result)
                        platform_counts[item['platform']] += 1
                        type_counts[item['type']] += 1
            
            remaining_slots = max_recommendations - len(recommendations)
            if remaining_slots > 0:
                used_ids = {rec['id'] for rec in recommendations}
                remaining_items = [item for item in similar_items if item['id'] not in used_ids]
                
                if cross_platform_only:
                    remaining_items = [item for item in remaining_items if item['platform'] != source_content['platform']]
                
                for item in remaining_items:

                    if platform_diversity and platform_counts[item['platform']] >= max_per_platform[item['platform']]:
                        continue
                        
                    if type_counts[item['type']] > max_recommendations * 0.75:
                        continue
                    
                    type_boost = 1.25 if (same_type_boost and 
                                    item['type'] == source_content['type']) else 1.0
                    
                    final_score = item['hybrid_score'] * type_boost * self.platform_diversity_weights[item['platform']]
                    
                    result = {
                        'id': item['id'],
                        'title': item['title'],
                        'platform': item['platform'],
                        'type': item['type'],
                        'score': final_score,
                        'genres': item['genres'],
                        'cluster': item['cluster'],
                        'rating_category': item.get('rating_category', 'Unknown'),
                        'split': item.get('split', 'unknown')
                    }
                    
                    if include_metrics:
                        result['metrics'] = {
                            'content_similarity': item['content_similarity'],
                            'genre_similarity': item['genre_similarity'],
                            'cluster_similarity': item['cluster_similarity'],
                            'thematic_similarity': item['thematic_similarity'],
                            'rating_compatibility': item['rating_compatibility']
                        }
                    
                    recommendations.append(result)
                    platform_counts[item['platform']] += 1
                    type_counts[item['type']] += 1
                    
                    if (len(recommendations) >= max_recommendations and 
                        min(platform_counts.values()) > 0):
                        break
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            recommendations = recommendations[:max_recommendations]
            
            return {
                'source_content': {
                    'id': source_content['id'],
                    'title': source_content['title'],
                    'platform': source_content['platform'],
                    'type': source_content['type'],
                    'genres': source_content['genres'],
                    'cluster': source_content['cluster'],
                    'rating_category': source_content.get('rating_category', 'Unknown'),
                    'split': source_content.get('split', 'unknown')
                },
                'recommendations': recommendations,
                'platform_distribution': platform_counts,
                'type_distribution': type_counts
            }

    def evaluate_model(self, sample_size: int = 50) -> Dict:
        if not self.processed_data:
            raise Exception('Model not initialized. Load data first.')
            
        logger.info(f"Evaluating content-based model with sample size {sample_size}")
        
        # Use validation data for evaluation
        val_items = [item for item in self.processed_data if item.get('split') == 'val']
        
        if len(val_items) == 0:
            logger.warning("No validation items found, using test items instead")
            val_items = [item for item in self.processed_data if item.get('split') == 'test']
            
        if len(val_items) == 0:
            logger.warning("No test items found either, using a random sample from all data")
            val_items = self.processed_data

        platform_samples = {}
        per_platform = min(sample_size // 3, 10)
        
        for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
            platform_items = [item for item in val_items if item['platform'] == platform]
            
            if platform_items:
                if len(platform_items) <= per_platform:
                    platform_samples[platform] = platform_items
                else:
                    np.random.seed(self.random_state)
                    platform_samples[platform] = np.random.choice(platform_items, size=per_platform, replace=False).tolist()
        
        sample_items = []
        for items in platform_samples.values():
            sample_items.extend(items)
        sample_items = sample_items[:sample_size]
        
        total_content_sim = 0
        total_cluster_sim = 0
        total_genre_sim = 0
        total_thematic_sim = 0
        successful_recs = 0
        cross_platform_recs = 0
        
        cluster_hits = {i: 0 for i in range(self.n_clusters)}
        platform_distribution = {'Netflix': 0, 'Amazon Prime': 0, 'Hulu': 0}
        
        for item in sample_items:
            try:
                recs = self.get_recommendations(
                    item['id'], 
                    {'max_recommendations': 5, 'platform_diversity': True, 'strict_type_matching': True}
                )
                
                if 'recommendations' in recs and recs['recommendations']:
                    # Count recommendations by platform
                    for rec in recs['recommendations']:
                        platform_distribution[rec['platform']] += 1
                        if rec['platform'] != item['platform']:
                            cross_platform_recs += 1
                    
                    # Calculate average similarity metrics
                    metrics = ['content_similarity', 'cluster_similarity', 
                             'genre_similarity', 'thematic_similarity']
                    avg_metrics = {
                        m: sum(r['metrics'][m] for r in recs['recommendations']) / len(recs['recommendations'])
                        for m in metrics
                    }
                    
                    total_content_sim += avg_metrics['content_similarity']
                    total_cluster_sim += avg_metrics['cluster_similarity']
                    total_genre_sim += avg_metrics['genre_similarity']
                    total_thematic_sim += avg_metrics['thematic_similarity']
                    successful_recs += 1
                    
                    for rec in recs['recommendations']:
                        if 'cluster' in rec and rec['cluster'] >= 0:
                            cluster_hits[rec['cluster']] += 1
            except Exception as e:
                logger.warning(f"Error evaluating item {item['id']}: {str(e)}")
        
        if successful_recs == 0:
            return {
                'model_type': 'content-based',
                'error': 'No successful recommendations'
            }
    
        avg_content_sim = total_content_sim / successful_recs
        avg_cluster_sim = total_cluster_sim / successful_recs
        avg_genre_sim = total_genre_sim / successful_recs
        avg_thematic_sim = total_thematic_sim / successful_recs
        
        total_recs = sum(platform_distribution.values())
        platform_percentages = {
            platform: (count / total_recs * 100) if total_recs > 0 else 0
            for platform, count in platform_distribution.items()
        }
        
        cross_platform_pct = (cross_platform_recs / total_recs * 100) if total_recs > 0 else 0
        
        # Calculate cluster coverage
        active_clusters = sum(1 for count in cluster_hits.values() if count > 0)
        cluster_coverage = active_clusters / self.n_clusters
        
        return {
            'model_type': 'content-based',
            'average_content_similarity': avg_content_sim,
            'average_cluster_similarity': avg_cluster_sim,
            'average_genre_similarity': avg_genre_sim,
            'average_thematic_similarity': avg_thematic_sim,
            'overall_score': (avg_content_sim + avg_cluster_sim + avg_genre_sim + avg_thematic_sim) / 4,
            'sample_size': sample_size,
            'successful_evaluations': successful_recs,
            'cross_platform_percentage': cross_platform_pct,
            'platform_distribution': platform_percentages,
            'cluster_coverage': cluster_coverage,
            'active_clusters': active_clusters
        }

    def get_content_diversity(self) -> Dict:
        if not self.processed_data:
            raise Exception('Model not initialized. Load data first.')
        
        platforms = ['Netflix', 'Amazon Prime', 'Hulu']
        results = {}
        
        for platform in platforms:
            platform_content = [item for item in self.processed_data 
                              if item['platform'] == platform]
            
            if not platform_content:
                continue
                
            # Genre diversity
            all_genres = set()
            total_genres = 0
            for item in platform_content:
                all_genres.update(item['genres'])
                total_genres += len(item['genres'])
            
            # Rating diversity
            rating_categories = ['Kids', 'Family', 'Teen', 'Adult', 'Unknown']
            rating_counts = {cat: 0 for cat in rating_categories}
            for item in platform_content:
                rating = item.get('rating_category', 'Unknown')
                if rating in rating_counts:
                    rating_counts[rating] += 1
            
            # Cluster diversity
            clusters = [item['cluster'] for item in platform_content]
            unique_clusters = len(set(clusters))
            cluster_entropy = 0
            if unique_clusters > 1:
                cluster_counts = np.bincount(clusters)
                cluster_probs = cluster_counts[cluster_counts > 0] / len(clusters)
                cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10))
            
            # Content type diversity
            movie_count = sum(1 for item in platform_content 
                            if item['content_features'].get('is_movie') == 1)
            tv_show_count = len(platform_content) - movie_count
            
            results[platform] = {
                'total_titles': len(platform_content),
                'unique_genres': len(all_genres),
                'genres_per_title': total_genres / len(platform_content),
                'rating_distribution': rating_counts,
                'movie_percentage': (movie_count / len(platform_content)) * 100,
                'tv_show_percentage': (tv_show_count / len(platform_content)) * 100,
                'unique_clusters': unique_clusters,
                'cluster_entropy': cluster_entropy,
                'cluster_coverage': unique_clusters / self.n_clusters
            }
        
        return results

    def cross_platform_eval(self) -> Dict:
        if not self.processed_data:
            raise Exception('Model not initialized. Load data first.')
        
        logger.info("Evaluating cross-platform recommendation performance")

        test_items = [item for item in self.processed_data if item.get('split') == 'test']
        
        if len(test_items) == 0:
            logger.warning("No test items found, using validation items instead")
            test_items = [item for item in self.processed_data if item.get('split') == 'val']
    
        platforms = ['Netflix', 'Amazon Prime', 'Hulu']
        platform_samples = {}
        sample_size = min(50, len(test_items) // 3)
        
        for platform in platforms:
            platform_items = [item for item in test_items if item['platform'] == platform]
            if platform_items:
                platform_samples[platform] = np.random.choice(platform_items, 
                                                            size=min(sample_size, len(platform_items)), 
                                                            replace=False).tolist()
        
        # Metrics to track
        metrics = {
            'total_samples': 0,
            'successful_recommendations': 0,
            'recommendations_by_platform': {p: 0 for p in platforms},
            'by_source_platform': {},
            'by_target_platform': {},
            'overall_similarity': 0.0
        }
        
        # For each platform, get cross-platform recommendations
        for source_platform, sample_items in platform_samples.items():
            metrics['by_source_platform'][source_platform] = {
                'samples': len(sample_items),
                'successful': 0,
                'target_platforms': {p: 0 for p in platforms if p != source_platform}
            }
            
            for item in sample_items:
                metrics['total_samples'] += 1
                
                try:
                    recs = self.get_recommendations(
                        item['id'],
                        options={
                            'max_recommendations': 5,
                            'cross_platform_only': True,
                            'platform_diversity': True
                        }
                    )
                    
                    if 'recommendations' in recs and recs['recommendations']:
                        metrics['successful_recommendations'] += 1
                        metrics['by_source_platform'][source_platform]['successful'] += 1
                        
                        for rec in recs['recommendations']:
                            target_platform = rec['platform']
                            metrics['recommendations_by_platform'][target_platform] += 1
                            
                            if target_platform != source_platform:
                                metrics['by_source_platform'][source_platform]['target_platforms'][target_platform] += 1
                            
                            if 'metrics' in rec:
                                metrics['overall_similarity'] += rec['metrics']['content_similarity']
                
                except Exception as e:
                    logger.warning(f"Error getting cross-platform recommendations for {item['id']}: {str(e)}")
        
        if metrics['successful_recommendations'] > 0:
            metrics['avg_similarity'] = metrics['overall_similarity'] / metrics['successful_recommendations']
            metrics['success_rate'] = metrics['successful_recommendations'] / metrics['total_samples'] * 100
            
            total_recs = sum(metrics['recommendations_by_platform'].values())
            metrics['platform_distribution_pct'] = {
                p: count / total_recs * 100 if total_recs > 0 else 0
                for p, count in metrics['recommendations_by_platform'].items()
            }
            
            for source, data in metrics['by_source_platform'].items():
                if data['samples'] > 0:
                    data['success_rate'] = data['successful'] / data['samples'] * 100
                    
                    total_targets = sum(data['target_platforms'].values())
                    if total_targets > 0:
                        data['target_distribution'] = {
                            target: count / total_targets * 100
                            for target, count in data['target_platforms'].items()
                        }
        
        return metrics

    def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        if not query or not self.processed_data:
            return []
        
        lower_query = query.lower()
        results = []
        
        exact_matches = []
        for item in self.processed_data:
            if lower_query in item['title'].lower():
                exact_matches.append({
                    'id': item['id'],
                    'title': item['title'],
                    'platform': item['platform'],
                    'type': item['type'],
                    'genres': item['genres'],
                    'cluster': item['cluster'],
                    'match_quality': 1.0,
                    'split': item.get('split', 'unknown') 
                })
        
        # If not enough exact matches, do fuzzy matching
        if len(exact_matches) < limit:
            query_tokens = set(lower_query.split())
            
            for item in self.processed_data:
                if any(m['id'] == item['id'] for m in exact_matches):
                    continue
                
                title_tokens = set(item['title'].lower().split())
                common_tokens = query_tokens.intersection(title_tokens)
                if common_tokens:
                    match_quality = len(common_tokens) / max(len(query_tokens), len(title_tokens))
                    if match_quality > 0.3:
                        results.append({
                            'id': item['id'],
                            'title': item['title'],
                            'platform': item['platform'],
                            'type': item['type'],
                            'genres': item['genres'],
                            'cluster': item['cluster'],
                            'match_quality': match_quality,
                            'split': item.get('split', 'unknown')  
                        })
        
        all_results = exact_matches + results
        all_results.sort(key=lambda x: x['match_quality'], reverse=True)
        
        return all_results[:limit]
    
    def find_similar_items(self, content_id: str, max_items: int = 50, diversity_factor: float = 0.5, recursion_depth: int = 0) -> List[Dict]:
        if recursion_depth >= 3:
            logger.warning(f"Maximum recursion depth reached for content_id {content_id}")
            return []  # 
        
        if not self.similarity_model:
            raise Exception("Similarity model not trained")
        
        source_item = None
        for i, item in enumerate(self.processed_data):
            if item['id'] == content_id:
                source_item = item
                break
        
        if not source_item:
            raise Exception(f"Content with ID {content_id} not found")
        
        feature_values = []
        for feature in self.similarity_model['features']:
            value = source_item['content_features'].get(feature, 0)
            feature_values.append(value)
        
        features_array = np.array([feature_values])
        scaler = self.similarity_model['scaler']
        features_scaled = scaler.transform(features_array)
        
        nn_model = self.similarity_model['model']
        neighbors_to_fetch = min(500, len(self.processed_data))  
        distances, indices = nn_model.kneighbors(features_scaled, n_neighbors=neighbors_to_fetch)
        
        training_indices = self.similarity_model['indices']
        actual_indices = [training_indices[idx] for idx in indices[0]]
        
        source_platform = source_item['platform']
        source_genres = source_item['genres']
        source_cluster = source_item['cluster']
        source_rating = source_item.get('rating_category', 'Unknown')
        source_type = source_item['type']
        source_title = source_item['title'].lower()
        
        similar_items = []
        platform_counts = {'Netflix': 0, 'Amazon Prime': 0, 'Hulu': 0}
        type_counts = {'Movie': 0, 'TV Show': 0}
        skipped_count = 0
        
        target_per_platform = max_items // 3
        min_per_platform = max(1, max_items // 5)
        
        for i, idx in enumerate(actual_indices):

            item_to_check = None
            for item in self.processed_data:
                if self.features_df.loc[idx, 'show_id'] == item['id']:
                    item_to_check = item
                    break
            
            if not item_to_check or item_to_check['id'] == source_item['id']:
                continue
                
            distance = distances[0][i]
            nn_similarity = np.exp(-distance)
            
            item_rating = item_to_check.get('rating_category', 'Unknown')
            rating_compat = self.calculate_rating_compatibility(source_rating, item_rating)
            rating_threshold = max(0.1, 0.3 - (recursion_depth * 0.1))
            if rating_compat <= rating_threshold:
                skipped_count += 1
                continue
            
            content_sim = self.calculate_content_similarity(source_item, item_to_check)
            genre_sim = self.calculate_genre_similarity(source_genres, item_to_check['genres'])
            cluster_sim = self.calculate_cluster_similarity(source_cluster, item_to_check['cluster'])
            thematic_sim = self.calculate_thematic_similarity(source_item, item_to_check)
            
            title_sim = self._calculate_enhanced_title_similarity(source_title, item_to_check['title'].lower())
            
            content_threshold = max(0.05, 0.1 - (recursion_depth * 0.03))
            genre_threshold = max(0.05, 0.15 - (recursion_depth * 0.05))
            if content_sim < content_threshold and genre_sim < genre_threshold:
                skipped_count += 1
                continue
            
            weights = self.recommendation_weights
            hybrid_score = (
                weights['content_weight'] * content_sim +
                weights['genre_weight'] * genre_sim +
                weights['cluster_weight'] * cluster_sim +
                weights['thematic_weight'] * thematic_sim
            ) / sum(weights.values())
            
            if title_sim > 0.3:
                hybrid_score *= (1.0 + title_sim * 0.3)
            
            hybrid_score *= rating_compat
            
            platform_bonus = 0
            current_platform = item_to_check['platform']
            
            if platform_counts[current_platform] < min_per_platform:
                # Big boost for underrepresented platforms
                platform_bonus = 0.3
            elif platform_counts[current_platform] < target_per_platform:
                # Moderate boost for platforms below target but not severely underrepresented
                platform_bonus = 0.15
            elif platform_counts[current_platform] >= target_per_platform * 1.5:
                # Penalty for overrepresented platforms
                platform_bonus = -0.2
            
            type_bonus = 0.15 if item_to_check['type'] == source_type else -0.05
            
            adjusted_score = (hybrid_score * (1 - diversity_factor) + 
                            (platform_bonus + type_bonus) * diversity_factor)

            if self._is_same_franchise(source_title, item_to_check['title'].lower()):
                adjusted_score *= 1.25
            
            similar_items.append({
                'id': item_to_check['id'],
                'title': item_to_check['title'],
                'platform': item_to_check['platform'],
                'type': item_to_check['type'],
                'genres': item_to_check['genres'],
                'cluster': item_to_check['cluster'],
                'rating': item_to_check.get('rating', 'Unknown'),
                'rating_category': item_rating,
                'distance': distance,
                'nn_similarity': nn_similarity,
                'content_similarity': content_sim,
                'genre_similarity': genre_sim,
                'cluster_similarity': cluster_sim,
                'thematic_similarity': thematic_sim,
                'title_similarity': title_sim, 
                'hybrid_score': hybrid_score,
                'adjusted_score': adjusted_score,
                'rating_compatibility': rating_compat,
                'split': item_to_check.get('split', 'unknown')
            })
            
            platform_counts[item_to_check['platform']] += 1
            type_counts[item_to_check['type']] += 1
        
        min_results = max(max_items // 2, 10)
        if len(similar_items) < min_results and skipped_count > 0 and recursion_depth < 2:
            logger.info(f"Relaxing filters (found {len(similar_items)} items, skipped {skipped_count})")
            new_diversity_factor = diversity_factor * 0.5
            additional_items = self.find_similar_items(
                content_id, 
                max_items=max_items, 
                diversity_factor=new_diversity_factor,
                recursion_depth=recursion_depth + 1
            )
            

            existing_ids = {item['id'] for item in similar_items}
            for item in additional_items:
                if item['id'] not in existing_ids:
                    similar_items.append(item)
                    existing_ids.add(item['id'])
        similar_items.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        final_selections = []
        remaining_items = []
        platform_selections = {'Netflix': 0, 'Amazon Prime': 0, 'Hulu': 0}
        
        for item in similar_items:
            platform = item['platform']
            if platform_selections[platform] < min_per_platform:
                final_selections.append(item)
                platform_selections[platform] += 1
            else:
                remaining_items.append(item)
        
        max_per_platform = max(target_per_platform, min_per_platform + 2)
        remaining_slots = max_items - len(final_selections)
        
        for item in remaining_items:
            if len(final_selections) >= max_items:
                break
                
            platform = item['platform']
            if platform_selections[platform] < max_per_platform:
                final_selections.append(item)
                platform_selections[platform] += 1
        
        if len(final_selections) < max_items:
            for item in remaining_items:
                if item not in final_selections and len(final_selections) < max_items:
                    final_selections.append(item)
        
        final_selections.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return final_selections[:max_items]

    def _calculate_enhanced_title_similarity(self, title1: str, title2: str) -> float:

        words1 = set(re.findall(r'\b\w+\b', title1))
        words2 = set(re.findall(r'\b\w+\b', title2))
        
        if not words1 or not words2:
            return 0.0
        
        common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        substring_match = 0.0
        if len(title1) > 5 and len(title2) > 5:
            if title1 in title2 or title2 in title1:
                substring_match = 0.8
        
        def get_ngrams(text, n=3):
            return [text[i:i+n] for i in range(len(text)-n+1)]
        
        if len(title1) >= 3 and len(title2) >= 3:
            ngrams1 = set(get_ngrams(title1))
            ngrams2 = set(get_ngrams(title2))
            
            ngram_intersection = ngrams1.intersection(ngrams2)
            ngram_union = ngrams1.union(ngrams2)
            ngram_sim = len(ngram_intersection) / len(ngram_union) if ngram_union else 0.0
        else:
            ngram_sim = 0.0
        
        # Calculate combined similarity
        similarity = max(jaccard, substring_match, ngram_sim)
        return similarity

    def _is_same_franchise(self, title1: str, title2: str) -> bool:
        franchise_indicators = [
            r'(?i)(.*?)\s*[:]\s*(.*)', 
            r'(?i)(.*?)\s+\d+\s*$',     
            r'(?i)(.*?)\s+part\s+\d+',  
            r'(?i)(.*?)\s+chapter\s+\d+', 
            r'(?i)(.*?)\s+the\s+\w+\s+\w+',  
        ]
        
        def extract_franchise(title):
            for pattern in franchise_indicators:
                match = re.match(pattern, title)
                if match:
                    return match.group(1).strip().lower()
            return None
        
        franchise1 = extract_franchise(title1)
        franchise2 = extract_franchise(title2)
        
        if franchise1 and franchise2 and franchise1 == franchise2:
            return True
        
        roman_pattern = r'(?i)(.*?)\s+(I{1,3}|IV|V|VI{1,3}|IX|X)$'
        
        match1 = re.match(roman_pattern, title1)
        match2 = re.match(roman_pattern, title2)
        
        if match1 and match2 and match1.group(1).strip().lower() == match2.group(1).strip().lower():
            return True
        
        # Check for sequential titles (e.g., "Movie Name", "Movie Name 2")
        base_title_pattern = r'(?i)(.*?)(?:\s+\d+|\s+part\s+\d+|\s+chapter\s+\d+|\s+the\s+\w+|\s+returns|\s+begins|$)'
        
        base1 = re.match(base_title_pattern, title1)
        base2 = re.match(base_title_pattern, title2)
        
        if base1 and base2:
            base1 = base1.group(1).strip().lower()
            base2 = base2.group(1).strip().lower()
            
            if base1 and base2 and (base1 == base2 or (len(base1) > 4 and len(base2) > 4 and 
                                                    (base1 in base2 or base2 in base1))):
                return True
        
        return False

    
    def train_and_compare_models(self):
        if self.train_data is None or self.val_data is None:
            raise ValueError("No train/val splits found. Make sure to load and preprocess data first.")

        target_col = "platform_encoded"
        if target_col not in self.train_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in train/val data.")
        
        numeric_features = [col for col in self.content_features if col in self.train_data.columns]
        
        # X_train, y_train
        X_train = self.train_data[numeric_features].fillna(0)
        y_train = self.train_data[target_col]

        # X_val, y_val
        X_val = self.val_data[numeric_features].fillna(0)
        y_val = self.val_data[target_col]

        models = {
            "LogisticRegression": LogisticRegression(
                random_state=self.random_state,
                max_iter=500
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state
            ),
            "KMeans": KMeans(
                n_clusters=len(pd.unique(y_train)),  
                random_state=self.random_state
            )
        }

        metrics_comparison = []

        for model_name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict on validation
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            metrics_comparison.append({
                "model": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
        
        print("Model Comparison (Validation Set):")
        print("--------------------------------------------------------")
        for entry in metrics_comparison:
            print(
                f"{entry['model']}: "
                f"Accuracy={entry['accuracy']:.3f}, "
                f"Precision={entry['precision']:.3f}, "
                f"Recall={entry['recall']:.3f}, "
                f"F1={entry['f1_score']:.3f}"
            )
        
        return metrics_comparison

#-------------------------------------------------
# Function to demo the recommendation system
#-------------------------------------------------

def demo_recommendations(model_path=None):
    print("\n=== Cross-Platform Recommendation Model Demo ===\n")
   
    model = ContentStreamingRecommendationModel()
    
    if model_path and os.path.exists(model_path):
   
        print(f"Loading model from {model_path}...")
        model.load_model(model_path)
        
        if not model.processed_data and os.path.exists("Data/train_data.csv"):
            print("Loading data from Data/ directory...")
            model.load_data("Data/train_data.csv")
    else:
        print("Training a new model...")
        if os.path.exists("Data/engineered_features.csv"):
            model.load_data("Data/engineered_features.csv")
        else:
            raise FileNotFoundError("Could not find engineered_features.csv")
    
    print("\nEvaluating model performance...")
    eval_results = model.evaluate_model(sample_size=30)
    print(f"Overall recommendation score: {eval_results['overall_score']:.4f}")
    print(f"Content similarity: {eval_results['average_content_similarity']:.4f}")
    print(f"Genre similarity: {eval_results['average_genre_similarity']:.4f}")
    print(f"Cross-platform recommendations: {eval_results['cross_platform_percentage']:.2f}%")
    
    print("\nTesting cross-platform recommendations...")
    cross_eval = model.cross_platform_eval()
    print(f"Success rate: {cross_eval.get('success_rate', 0):.2f}%")
    print(f"Average similarity: {cross_eval.get('avg_similarity', 0):.4f}")
    print("Platform distribution:")
    for platform, pct in cross_eval.get('platform_distribution_pct', {}).items():
        print(f"  {platform}: {pct:.2f}%")
    
    print("\nSample recommendations:")
    sample_queries = ["Stranger Things", "The Avengers", "The Handmaid's Tale"]
    
    for query in sample_queries:
        print(f"\nSearching for: '{query}'")
        search_results = model.search_content(query, limit=1)
        
        if not search_results:
            print(f"  No matches found for '{query}'")
            continue
            
        item = search_results[0]
        print(f"  Found: '{item['title']}' ({item['type']}) on {item['platform']}")
        
        print(f"  Getting recommendations across all platforms...")
        recs = model.get_recommendations(
            item['id'],
            options={
                'max_recommendations': 3,
                'cross_platform_only': False,
                'platform_diversity': True
            }
        )
        
        if 'recommendations' in recs and recs['recommendations']:
            print(f"  Top recommendations:")
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"    {i}. '{rec['title']}' ({rec['type']}) on {rec['platform']} - Score: {rec['score']:.4f}")
        else:
            print(f"  No recommendations found")
    
    if not model_path:
        save_path = "Models/cross_platform_recommender.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        print(f"\nModel saved to {save_path}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_recommendations()
    model = ContentStreamingRecommendationModel()
    model.load_data("Data/test_data.csv")  # or whichever CSV you use
    comparison_results = model.train_and_compare_models()