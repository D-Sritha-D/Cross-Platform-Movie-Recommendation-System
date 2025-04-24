import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def content_based_recommendation_system():
    print("Loading data for content-based recommendations...")

    try:
        print("Loading dataset...")
        data_file = 'Data/all_platforms_combined.csv'
        if not os.path.exists(data_file):
            data_file = 'all_platforms_combined.csv'
            if not os.path.exists(data_file):
                print("Error: Could not find dataset file.")
                return
        
        df = pd.read_csv(data_file)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        df['content_features'] = ''
        if 'platform' in df.columns:
            df['content_features'] += df['platform'].fillna('') + ' '
        
        if 'type' in df.columns:
            df['content_features'] += df['type'].fillna('') + ' '
        
        if 'listed_in' in df.columns:
            df['content_features'] += df['listed_in'].fillna('') + ' '
        
        if 'description' in df.columns:
            df['content_features'] += df['description'].fillna('') + ' '
            
        if 'director' in df.columns:
            df['content_features'] += df['director'].fillna('') + ' '
        
        if 'cast' in df.columns:
            df['content_features'] += df['cast'].fillna('') + ' '
        
        df['content_features'] = df['content_features'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower()))
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['content_features'])
        
        def get_similar_titles(title, n_recommendations=5):
            if title not in df['title'].values:
                print(f"Title '{title}' not found in the dataset.")
                similar_titles = df[df['title'].str.contains(title, case=False, na=False)]
                if len(similar_titles) > 0:
                    title = similar_titles.iloc[0]['title']
                    print(f"Using closest match: '{title}'")
                else:
                    return pd.DataFrame()
            
            idx = df[df['title'] == title].index[0]
            similarity_scores = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
            similar_indices = similarity_scores.argsort()[-n_recommendations-1:-1][::-1]
            
            similar_df = df.iloc[similar_indices][['title', 'platform', 'type']]
            if 'release_year' in df.columns:
                similar_df['release_year'] = df.iloc[similar_indices]['release_year']
            if 'rating' in df.columns:
                similar_df['rating'] = df.iloc[similar_indices]['rating']
            if 'listed_in' in df.columns:
                similar_df['genres'] = df.iloc[similar_indices]['listed_in']
            similar_df['similarity_score'] = similarity_scores[similar_indices]
            
            return similar_df
        
        # Main recommendation function
        def recommend_from_samples():
            print("\n===== CONTENT-BASED RECOMMENDATION SYSTEM =====")
            print("\nHere are some sample titles from the dataset:")
            
            # Get a stratified sample of titles
            sample_size = 5
            sample_titles = []
            
            if 'platform' in df.columns:
                for platform in df['platform'].unique():
                    platform_df = df[df['platform'] == platform]
                    if 'type' in df.columns:
                        for content_type in platform_df['type'].unique():
                            type_df = platform_df[platform_df['type'] == content_type]
                            if len(type_df) > 0:
                                sample_titles.append(type_df.sample(1)['title'].iloc[0])
                            if len(sample_titles) >= sample_size:
                                break
                    else:
                        if len(platform_df) > 0:
                            sample_titles.append(platform_df.sample(1)['title'].iloc[0])
                    if len(sample_titles) >= sample_size:
                        break
            
            if len(sample_titles) < sample_size:
                additional_samples = df.sample(sample_size - len(sample_titles))['title'].tolist()
                sample_titles.extend(additional_samples)
            
            for i, title in enumerate(sample_titles, 1):
                print(f"{i}. {title}")
            
            print("\nEnter the number of a sample title, or type your own title:")
            user_input = input()
            
            selected_title = ""
            if user_input.isdigit() and 1 <= int(user_input) <= len(sample_titles):
                selected_title = sample_titles[int(user_input) - 1]
            else:
                selected_title = user_input
            
            print(f"\nFinding recommendations similar to: '{selected_title}'")
            
            recommendations = get_similar_titles(selected_title)
            
            if len(recommendations) == 0:
                print("Could not find recommendations for that title.")
                retry = input("Would you like to try again with a different title? (y/n): ")
                if retry.lower() == 'y':
                    recommend_from_samples()
                return
            
            print("\n===== RECOMMENDED TITLES =====")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(recommendations)
            
            print("\nWhy these recommendations?")
            print("These titles were recommended because they share similar features with your selected title, such as:")
            print("- Same platform or content type")
            print("- Similar genres or categories")
            print("- Similar descriptions, themes, or keywords")
            print("- Same directors or cast members (if available)")
            print("- The similarity score shows how close each recommendation is to your selected title")
            
            more = input("\nWould you like recommendations for another title? (y/n): ")
            if more.lower() == 'y':
                recommend_from_samples()
        recommend_from_samples()
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    content_based_recommendation_system()