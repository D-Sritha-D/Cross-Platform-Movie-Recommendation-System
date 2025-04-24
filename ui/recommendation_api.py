from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Global variables to store loaded data and models
df = None
tfidf_matrix = None
content_features = None
xgboost_model = None
xgb_label_encoder = None

def load_data():
    global df, tfidf_matrix, content_features, xgboost_model, xgb_label_encoder
    
    print("Loading dataset and models...")
    
    data_files = [
        'Data/all_platforms_combined.csv',
        'all_platforms_combined.csv',
        'processed_streaming_data.csv',
        'Data/processed_streaming_data.csv',
        'selected_features_for_recommendation.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            break
    
    if df is None:
        print("Error: Could not find any dataset file.")
        return False
    
    print("Loading XGBoost model...")
    model_path = 'Models/XGBoost.pkl'
    encoder_path = 'Models/XGBoost_label_encoder.pkl'
    
    if not os.path.exists(model_path):
        model_path = 'Models/XGBoost_tuned.pkl'
        encoder_path = 'Models/XGBoost_tuned_label_encoder.pkl'
    
    if os.path.exists(model_path):
        try:
            xgboost_model = joblib.load(model_path)
            print(f"Successfully loaded XGBoost model from {model_path}")
            
            if os.path.exists(encoder_path):
                xgb_label_encoder = joblib.load(encoder_path)
                print(f"Successfully loaded label encoder from {encoder_path}")
        except Exception as e:
            print(f"Error loading XGBoost model: {str(e)}")
            return False
    else:
        print(f"Error: XGBoost model not found at {model_path}")
        return False
    
    print("Preparing content features...")
    
    df['content_features'] = ''
    
    if 'platform' in df.columns:
        df['content_features'] += df['platform'].fillna('') + ' '
    
    if 'type' in df.columns:
        df['content_features'] += df['type'].fillna('') + ' '
    
    if 'listed_in' in df.columns:
        df['content_features'] += df['listed_in'].fillna('') + ' '
    elif 'genre_list' in df.columns:
        df['content_features'] += df['genre_list'].fillna('') + ' '
    
    if 'description' in df.columns:
        df['content_features'] += df['description'].fillna('') + ' '
        
    if 'director' in df.columns:
        df['content_features'] += df['director'].fillna('') + ' '
    
    if 'cast' in df.columns:
        df['content_features'] += df['cast'].fillna('') + ' '
    
    df['content_features'] = df['content_features'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower())
    )
    
    print("Creating TF-IDF matrix...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    content_features = tfidf
    
    print("Data and models loaded successfully!")
    return True

@app.route('/api/platforms', methods=['GET'])
def get_platforms():
    """Get available streaming platforms"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    platforms = []
    if 'platform' in df.columns:
        platforms = df['platform'].unique().tolist()
    
    return jsonify({"platforms": platforms})

@app.route('/api/content_types', methods=['GET'])
def get_content_types():
    """Get available content types"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    content_types = []
    if 'type' in df.columns:
        content_types = df['type'].unique().tolist()
    
    return jsonify({"content_types": content_types})

@app.route('/api/sample_titles', methods=['GET'])
def get_sample_titles():
    """Get sample titles for the UI"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    platform = request.args.get('platform', 'All')
    content_type = request.args.get('type', 'All')
    
    filtered_df = df.copy()
    
    if platform != 'All' and 'platform' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    
    if content_type != 'All' and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    sample_size = min(10, len(filtered_df))
    if sample_size > 0:
        sample_df = filtered_df.sample(sample_size)
    else:
        return jsonify({"titles": []})
    
    titles = []
    for _, row in sample_df.iterrows():
        title_data = {
            "id": row.get('show_id', str(len(titles))),
            "title": row.get('title', '')
        }
        
        if 'platform' in row:
            title_data['platform'] = row['platform']
        if 'type' in row:
            title_data['type'] = row['type']
        if 'release_year' in row:
            title_data['year'] = int(row['release_year'])
        if 'rating' in row:
            title_data['rating'] = row['rating']
        if 'listed_in' in row:
            title_data['genres'] = row['listed_in']
        
        titles.append(title_data)
    
    return jsonify({"titles": titles})

@app.route('/api/search', methods=['GET'])
def search_titles():
    """Search for titles based on query"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    query = request.args.get('query', '')
    platform = request.args.get('platform', 'All')
    content_type = request.args.get('type', 'All')
    
    if not query:
        return jsonify({"results": []})
    
    filtered_df = df.copy()
    
    if platform != 'All' and 'platform' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    
    if content_type != 'All' and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    results_df = filtered_df[filtered_df['title'].str.contains(query, case=False, na=False)]
    
    results_df = results_df.head(20)
    
    results = []
    for _, row in results_df.iterrows():
        result_data = {
            "id": row.get('show_id', str(len(results))),
            "title": row.get('title', '')
        }
        
        if 'platform' in row:
            result_data['platform'] = row['platform']
        if 'type' in row:
            result_data['type'] = row['type']
        if 'release_year' in row:
            result_data['year'] = int(row['release_year'])
        if 'rating' in row:
            result_data['rating'] = row['rating']
        if 'listed_in' in row:
            result_data['genres'] = row['listed_in']
        
        results.append(result_data)
    
    return jsonify({"results": results})

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get recommendations using XGBoost and content similarity"""
    if df is None or tfidf_matrix is None or xgboost_model is None:
        return jsonify({"error": "Data or models not loaded"}), 500
    
    title_id = request.args.get('id')
    title_name = request.args.get('title')
    
    title_idx = None
    
    if title_id is not None:
        title_rows = df[df['show_id'] == title_id]
        if not title_rows.empty:
            title_idx = title_rows.index[0]
    
    if title_idx is None and title_name is not None:
        title_rows = df[df['title'] == title_name]
        if not title_rows.empty:
            title_idx = title_rows.index[0]
    
    if title_idx is None:
        return jsonify({"error": "Title not found"}), 404
    selected_title = df.iloc[title_idx]
    
    # Method 1: Use XGBoost model to get feature importance
    try:
        feature_importances = xgboost_model.feature_importances_
        top_feature_indices = feature_importances.argsort()[-10:][::-1]
    except:
        feature_columns = df.select_dtypes(include=['int64', 'float64']).columns
        top_feature_indices = range(len(feature_columns))
    
    # Method 2: Use content-based filtering with TF-IDF
    sim_scores = cosine_similarity(tfidf_matrix[title_idx:title_idx+1], tfidf_matrix).flatten()
    
    # Method 3: Hybrid approach - combine content similarity with XGBoost importance
    sim_indices = sim_scores.argsort()[-50:][::-1] 
    sim_indices = sim_indices[sim_indices != title_idx]
    hybrid_scores = []
    
    for idx in sim_indices:
        score = sim_scores[idx]
        feature_match_bonus = 0
        
        selected_row = df.iloc[title_idx]
        candidate_row = df.iloc[idx]
        
        if 'platform' in df.columns and selected_row['platform'] == candidate_row['platform']:
            feature_match_bonus += 0.1
            
        if 'type' in df.columns and selected_row['type'] == candidate_row['type']:
            feature_match_bonus += 0.1
            
        if 'release_year' in df.columns:
            year_diff = abs(selected_row['release_year'] - candidate_row['release_year'])
            if year_diff <= 3:
                feature_match_bonus += 0.1 * (1 - (year_diff / 10))
        
        if 'listed_in' in df.columns:
            selected_genres = str(selected_row['listed_in']).lower()
            candidate_genres = str(candidate_row['listed_in']).lower()
            matches = 0
            for genre in selected_genres.split(','):
                if genre.strip() in candidate_genres:
                    matches += 1
                    
            feature_match_bonus += min(0.2, 0.05 * matches)
            
        adjusted_score = score * (1 + feature_match_bonus)
        hybrid_scores.append((idx, adjusted_score))
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in hybrid_scores[:10]]
    
    recommendations = []
    for i, idx in enumerate(top_indices):
        row = df.iloc[idx]
        sim_score = sim_scores[idx] 
        hybrid_score = next(score for index, score in hybrid_scores if index == idx)
        
        rec = {
            "id": row.get('show_id', str(i)),
            "title": row.get('title', ''),
            "similarity": float(sim_score),
            "hybrid_score": float(hybrid_score)
        }

        if 'platform' in row:
            rec['platform'] = row['platform']
        if 'type' in row:
            rec['type'] = row['type']
        if 'release_year' in row:
            rec['year'] = int(row['release_year'])
        if 'rating' in row:
            rec['rating'] = row['rating']
        if 'listed_in' in row:
            rec['genres'] = row['listed_in']
        
        recommendations.append(rec)
    
    selected = {
        "id": selected_title.get('show_id', ''),
        "title": selected_title.get('title', '')
    }
    
    if 'platform' in selected_title:
        selected['platform'] = selected_title['platform']
    if 'type' in selected_title:
        selected['type'] = selected_title['type']
    if 'release_year' in selected_title:
        selected['year'] = int(selected_title['release_year'])
    if 'rating' in selected_title:
        selected['rating'] = selected_title['rating']
    if 'listed_in' in selected_title:
        selected['genres'] = selected_title['listed_in']
    
    return jsonify({
        "selected": selected,
        "recommendations": recommendations,
        "model": "XGBoost Hybrid"
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    stats = {
        "total_titles": len(df),
        "platforms": {},
        "content_types": {},
        "release_years": {},
        "model_info": {
            "name": "XGBoost Hybrid",
            "description": "A hybrid recommendation system using XGBoost feature importance and TF-IDF content similarity"
        }
    }
    
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts().to_dict()
        stats["platforms"] = platform_counts
    
    if 'type' in df.columns:
        type_counts = df['type'].value_counts().to_dict()
        stats["content_types"] = type_counts
    
    if 'release_year' in df.columns:
        df['decade'] = (df['release_year'] // 10) * 10
        decade_counts = df['decade'].value_counts().sort_index().to_dict()
        stats["release_decades"] = decade_counts
        
        current_year = df['release_year'].max()
        recent_years = df[df['release_year'] >= current_year - 10]['release_year'].value_counts().sort_index().to_dict()
        stats["recent_years"] = recent_years
    
    return jsonify(stats)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "XGBoost Streaming Content Recommendation API",
        "endpoints": [
            {"path": "/api/platforms", "description": "Get all available streaming platforms"},
            {"path": "/api/content_types", "description": "Get all available content types"},
            {"path": "/api/sample_titles", "description": "Get sample titles for the UI"},
            {"path": "/api/search", "description": "Search for titles based on query"},
            {"path": "/api/recommendations", "description": "Get hybrid recommendations using XGBoost and content similarity"},
            {"path": "/api/stats", "description": "Get dataset statistics"}
        ],
        "model": "XGBoost Hybrid"
    })

if __name__ == '__main__':
    if load_data():
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("Failed to load data or models. Exiting...")