import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def clean_datasets(netflix_df, amazon_df, hulu_df):
    dfs = [netflix_df, amazon_df, hulu_df]
    cleaned_dfs = []
    
    for df in dfs:
        df_clean = df.copy()
        
        # 1. Handle missing values
        for col in ['director', 'cast', 'country']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        if 'description' in df_clean.columns:
            df_clean['description'] = df_clean['description'].fillna('')
        if 'rating' in df_clean.columns:
            df_clean['rating'] = df_clean['rating'].fillna('Not Rated')
            
        # 2. Standardize date_added format
        if 'date_added' in df_clean.columns:
            df_clean['date_added'] = pd.to_datetime(df_clean['date_added'], errors='coerce')
        
        # 3. Clean duration field
        if 'duration' in df_clean.columns:
            df_clean['duration_numeric'] = df_clean['duration'].str.extract(r'(\d+)').astype('float')
            df_clean['duration_unit'] = df_clean['duration'].str.contains('min').map({True: 'min', False: 'season'})
            df_clean['duration_unit'] = df_clean['duration_unit'].fillna('unknown')
        
        # 4. Ensure release_year is numeric
        if 'release_year' in df_clean.columns:
            df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
            median_year = df_clean['release_year'].median()
            df_clean['release_year'] = df_clean['release_year'].fillna(median_year)
            df_clean['release_year'] = df_clean['release_year'].astype('int')
        
        # 5. Standardize type field
        if 'type' in df_clean.columns:
            type_mapping = {
                'movie': 'Movie',
                'movies': 'Movie',
                'film': 'Movie',
                'tv': 'TV Show',
                'tv show': 'TV Show',
                'tv_show': 'TV Show',
                'show': 'TV Show',
                'series': 'TV Show'
            }

            df_clean['type'] = df_clean['type'].str.lower()
            for key, value in type_mapping.items():
                df_clean.loc[df_clean['type'] == key, 'type'] = value
        
        # 6. Handle duplicates
        df_clean = df_clean.drop_duplicates(subset=['title', 'release_year'], keep='first')
        
        # 7. Standardize genre/category listings
        if 'listed_in' in df_clean.columns:
            df_clean['listed_in'] = df_clean['listed_in'].str.replace(' ,', ',').str.strip()
            genre_mapping = {
                'Comedies': 'Comedy', 
                'Comedy Movies': 'Comedy',
                'Stand-Up Comedy': 'Stand-Up',
                'Stand Up Comedy': 'Stand-Up',
                'Dramas': 'Drama',
                'Drama Movies': 'Drama',
                'Action & Adventure': 'Action',
                'Action and Adventure': 'Action',
                'Sci-Fi': 'Science Fiction',
                'SciFi': 'Science Fiction',
                'Science-Fiction': 'Science Fiction',
                'Documentaries': 'Documentary',
                'Documentary Films': 'Documentary',
                'Kids': "Children's",
                "Children": "Children's",
                'Crime Films': 'Crime',
                'Thriller Movies': 'Thriller'
            }
            
            for old, new in genre_mapping.items():
                df_clean['listed_in'] = df_clean['listed_in'].str.replace(old, new, regex=False)
        
        # 8. Clean country names
        if 'country' in df_clean.columns:
            us_variations = ['United States', 'USA', 'U.S.A.', 'US', 'U.S.', 'America', 'United States of America']
            for variation in us_variations:
                df_clean.loc[df_clean['country'].str.contains(variation, na=False, case=False), 'country'] = 'United States'
            uk_variations = ['United Kingdom', 'UK', 'U.K.', 'Britain', 'Great Britain', 'England']
            for variation in uk_variations:
                df_clean.loc[df_clean['country'].str.contains(variation, na=False, case=False), 'country'] = 'United Kingdom'
        
        # 9. Remove any HTML or special characters from text fields
        for col in ['title', 'description']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].str.replace('<.*?>', '', regex=True)
                df_clean[col] = df_clean[col].str.normalize('NFKD')
        
        # 10. Handle outliers in release_year
        if 'release_year' in df_clean.columns:
            current_year = datetime.now().year
            df_clean.loc[df_clean['release_year'] > current_year, 'release_year'] = current_year
            df_clean.loc[df_clean['release_year'] < 1900, 'release_year'] = np.nan
            df_clean['release_year'] = df_clean['release_year'].fillna(df_clean['release_year'].median())
            df_clean['release_year'] = df_clean['release_year'].astype('int')
        
        # 11. Standardize ratings across platforms
        if 'rating' in df_clean.columns:
            rating_mapping = {
                'TV-Y': 'TV-Y',
                'TV-Y7': 'TV-Y7',
                'TV-Y7-FV': 'TV-Y7',
                'G': 'G',
                'TV-G': 'TV-G',
                'PG': 'PG',
                'TV-PG': 'TV-PG',
                'PG-13': 'PG-13',
                'TV-14': 'TV-14',
                'R': 'R',
                'TV-MA': 'TV-MA',
                'NC-17': 'NC-17',
                'NR': 'Not Rated',
                'UR': 'Not Rated',
                'NOT RATED': 'Not Rated',
                'UNRATED': 'Not Rated',
                'ALL': 'G',
                'ALL_AGES': 'G',
                '13+': 'PG-13',
                '16+': 'TV-14',
                '18+': 'TV-MA',
                '7+': 'TV-Y7',
                'AGES_16_': 'TV-14',
                'AGES_18_': 'TV-MA',
                'NA': 'Not Rated',
                'nan': 'Not Rated'
            }
            df_clean['rating'] = df_clean['rating'].fillna('Not Rated')
            df_clean['rating'] = df_clean['rating'].replace(rating_mapping)
        
        # 12. Clean and standardize show_id format
        if 'show_id' in df_clean.columns:
            df_clean['show_id'] = df_clean['show_id'].astype(str)
            platform_name = None
            if 'platform' in df_clean.columns:
                sample_platform = df_clean['platform'].iloc[0] if not df_clean.empty else None
                if sample_platform == 'Netflix':
                    platform_name = 'nf'
                elif sample_platform == 'Amazon Prime':
                    platform_name = 'ap'
                elif sample_platform == 'Hulu':
                    platform_name = 'hu'
            
            if platform_name:
                mask = ~df_clean['show_id'].str.startswith(platform_name)
                df_clean.loc[mask, 'show_id'] = platform_name + '_' + df_clean.loc[mask, 'show_id']
        
        # 13. Handle inconsistent listed_in formats (make it a list where needed)
        if 'listed_in' in df_clean.columns:
            df_clean['listed_in'] = df_clean['listed_in'].fillna('Unknown')
            df_clean['listed_in'] = df_clean['listed_in'].astype(str)
            df_clean['genre_list'] = df_clean['listed_in'].str.split(',').apply(
                lambda x: [genre.strip() for genre in x if genre.strip()]
            )
            
        cleaned_dfs.append(df_clean)
    
    return cleaned_dfs[0], cleaned_dfs[1], cleaned_dfs[2]

def load_datasets(netflix_path, amazon_path, hulu_path):
    dataset_info = {
        "before_cleaning": {},
        "after_cleaning": {}
    }

    try:
        netflix_df = pd.read_csv(netflix_path, encoding='utf-8')
    except UnicodeDecodeError:
        netflix_df = pd.read_csv(netflix_path, encoding='latin1')
        
    try:
        amazon_df = pd.read_csv(amazon_path, encoding='utf-8')
    except UnicodeDecodeError:
        amazon_df = pd.read_csv(amazon_path, encoding='latin1')
        
    try:
        hulu_df = pd.read_csv(hulu_path, encoding='utf-8')
    except UnicodeDecodeError:
        hulu_df = pd.read_csv(hulu_path, encoding='latin1')
    
    # Store dataset information before cleaning so that the original data isn't disturbed
    dataset_info["before_cleaning"] = {
        "Netflix": {"rows": netflix_df.shape[0], "columns": netflix_df.shape[1]},
        "Amazon Prime": {"rows": amazon_df.shape[0], "columns": amazon_df.shape[1]},
        "Hulu": {"rows": hulu_df.shape[0], "columns": hulu_df.shape[1]}
    }
    
    netflix_df, amazon_df, hulu_df = clean_datasets(netflix_df, amazon_df, hulu_df)
    
    # Store dataset information after cleaning
    dataset_info["after_cleaning"] = {
        "Netflix": {"rows": netflix_df.shape[0], "columns": netflix_df.shape[1]},
        "Amazon Prime": {"rows": amazon_df.shape[0], "columns": amazon_df.shape[1]},
        "Hulu": {"rows": hulu_df.shape[0], "columns": hulu_df.shape[1]}
    }
    
    netflix_df['platform'] = 'Netflix'
    amazon_df['platform'] = 'Amazon Prime'
    hulu_df['platform'] = 'Hulu'
    all_platforms = pd.concat([netflix_df, amazon_df, hulu_df], ignore_index=True)
    
    required_columns = ['show_id', 'type', 'title', 'director', 'cast', 
                       'country', 'date_added', 'release_year', 
                       'rating', 'duration', 'listed_in', 'description', 'platform']
    
    for col in required_columns:
        if col not in all_platforms.columns:
            all_platforms[col] = None
    
    dataset_info["combined"] = {
        "rows": all_platforms.shape[0],
        "columns": all_platforms.shape[1]
    }
    
    return netflix_df, amazon_df, hulu_df, all_platforms, dataset_info

def get_basic_stats(df, platform_name):
    stats = {}
    stats['platform'] = platform_name
    stats['total_titles'] = len(df)
    stats['shape'] = df.shape
    stats['columns'] = df.columns.tolist()
    stats['missing_values'] = df.isnull().sum().to_dict()
    stats['content_type_counts'] = df['type'].value_counts().to_dict()
    
    # Year distribution
    years = df['release_year'].value_counts().sort_index()
    stats['release_year_range'] = (years.index.min(), years.index.max())
    stats['median_release_year'] = df['release_year'].median()
    
    # Ratings distribution
    stats['ratings_distribution'] = df['rating'].value_counts().to_dict()
    
    # Duplicate check
    stats['duplicate_titles'] = df['title'].duplicated().sum()
    
    # Content added per year (if date_added exists)
    if 'date_added' in df.columns:
        df['date_added_dt'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added_dt'].dt.year
        stats['additions_by_year'] = df['year_added'].value_counts().sort_index().to_dict()
    
    return stats

def analyze_content_types(all_platforms):
    results = {}
    
    # Content type counts by platform
    type_by_platform = pd.crosstab(all_platforms['platform'], all_platforms['type'])
    results['type_counts'] = type_by_platform.to_dict()
    
    type_pct = pd.crosstab(all_platforms['platform'], all_platforms['type'], 
                         normalize='index') * 100
    results['type_percentages'] = type_pct.round(2).to_dict()
    
    # Movie to TV Show ratio
    results['movie_to_tv_ratio'] = {}
    for platform in type_by_platform.index:
        if 'TV Show' in type_by_platform.columns and 'Movie' in type_by_platform.columns:
            tv_count = type_by_platform.loc[platform, 'TV Show']
            movie_count = type_by_platform.loc[platform, 'Movie']
            if tv_count > 0:
                results['movie_to_tv_ratio'][platform] = round(movie_count / tv_count, 2)
            else:
                results['movie_to_tv_ratio'][platform] = float('inf')  # Avoid division by zero
                
    # Content release trends over time
    release_trends = all_platforms.groupby(['platform', 'release_year']).size().unstack(level=0).fillna(0)
    results['release_trends'] = {year: {platform: int(count) for platform, count in year_data.items()} 
                               for year, year_data in release_trends.iterrows()}
    
    # Recent content (last 5 years)
    current_year = datetime.now().year
    recent_years = list(range(current_year-5, current_year+1))
    recent_content = all_platforms[all_platforms['release_year'].isin(recent_years)]
    results['recent_content_counts'] = recent_content.groupby('platform').size().to_dict()
    results['recent_content_percentage'] = ((recent_content.groupby('platform').size() / 
                                           all_platforms.groupby('platform').size()) * 100).round(2).to_dict()
    
    return results

def extract_genres(df):
    all_genres = []
    platform_genres = {}
    
    for platform in df['platform'].unique():
        platform_genres[platform] = []
    
    for _, row in df.iterrows():
        if pd.notna(row['listed_in']):
            genres = [genre.strip() for genre in row['listed_in'].split(',')]
            all_genres.extend(genres)
            platform_genres[row['platform']].extend(genres)
    
    return all_genres, platform_genres

def analyze_genres(all_platforms):

    results = {}
    all_genres, platform_genres = extract_genres(all_platforms)
    
    # Overall genre counts
    overall_genres = Counter(all_genres)
    results['top_genres_overall'] = dict(overall_genres.most_common(20))
    
    # Genre counts by platform
    results['top_genres_by_platform'] = {}
    for platform, genres in platform_genres.items():
        plt_genres = Counter(genres)
        results['top_genres_by_platform'][platform] = dict(plt_genres.most_common(10))
    
    # Genre comparison for top genres
    top_genres = [genre for genre, _ in overall_genres.most_common(15)]
    genre_comparison = {}
    
    for genre in top_genres:
        genre_comparison[genre] = {
            platform: Counter(platform_genres[platform])[genre]
            for platform in platform_genres.keys()
        }
    
    results['genre_comparison'] = genre_comparison
    
    # Unique genres by platform
    results['unique_genres'] = {}
    for platform, genres in platform_genres.items():
        unique_genres = set(genres)
        other_platforms = [p for p in platform_genres.keys() if p != platform]
        other_genres = set()
        for p in other_platforms:
            other_genres.update(platform_genres[p])
        
        exclusive_genres = unique_genres - other_genres
        results['unique_genres'][platform] = list(exclusive_genres)
    
    # Genre diversity (number of unique genres)
    results['genre_diversity'] = {platform: len(set(genres)) 
                                for platform, genres in platform_genres.items()}
    
    return results

def analyze_content_age(all_platforms):
    results = {}
    
    # Current year for reference
    current_year = datetime.now().year
    
    # Age of content calculation
    all_platforms['content_age'] = current_year - all_platforms['release_year']
    
    # Average content age by platform
    avg_age = all_platforms.groupby('platform')['content_age'].mean()
    results['average_content_age'] = avg_age.round(2).to_dict()
    
    # Median content age by platform
    median_age = all_platforms.groupby('platform')['content_age'].median()
    results['median_content_age'] = median_age.round(2).to_dict()
    
    # Age distribution metrics
    results['age_distribution'] = {}
    for platform in all_platforms['platform'].unique():
        platform_data = all_platforms[all_platforms['platform'] == platform]
        age_counts = platform_data['content_age'].value_counts().sort_index()
        results['age_distribution'][platform] = age_counts.to_dict()
    
    # Age buckets for analyzing the different ages of the content
    age_bins = [0, 2, 5, 10, 20, 50, float('inf')]
    age_labels = ['0-2 years', '3-5 years', '6-10 years', '11-20 years', '21-50 years', '50+ years']
    
    all_platforms['age_category'] = pd.cut(all_platforms['content_age'], 
                                         bins=age_bins, 
                                         labels=age_labels)
    
    age_category_counts = pd.crosstab(all_platforms['platform'], all_platforms['age_category'])
    results['age_categories'] = {
        platform: {category: int(count) for category, count in row.items()}
        for platform, row in age_category_counts.iterrows()
    }
    
    # Age category percentages
    age_category_pct = pd.crosstab(all_platforms['platform'], 
                                  all_platforms['age_category'], 
                                  normalize='index') * 100
    results['age_categories_pct'] = {
        platform: {category: round(pct, 2) for category, pct in row.items()}
        for platform, row in age_category_pct.iterrows()
    }
    
    # Content freshness (% of content less than 5 years old)
    fresh_content = all_platforms[all_platforms['content_age'] <= 5]
    freshness_pct = (fresh_content.groupby('platform').size() / 
                    all_platforms.groupby('platform').size() * 100)
    results['content_freshness_pct'] = freshness_pct.round(2).to_dict()
    
    return results

def standardize_rating(rating):
    if pd.isna(rating) or rating == 'NaN':
        return 'Unknown'
    
    # TV ratings
    if 'TV-Y' in str(rating):
        return 'TV-Y/TV-Y7 (Children)'
    elif 'TV-G' in str(rating):
        return 'TV-G (General Audience)'
    elif 'TV-PG' in str(rating):
        return 'TV-PG (Parental Guidance)'
    elif 'TV-14' in str(rating):
        return 'TV-14 (14+)'
    elif 'TV-MA' in str(rating):
        return 'TV-MA (Mature)'
            
    # Movie ratings
    elif rating in ['G', 'TV-G']:
        return 'G (General Audience)'
    elif rating in ['PG', 'TV-PG']:
        return 'PG (Parental Guidance)'
    elif rating in ['PG-13', 'TV-14']:
        return 'PG-13/TV-14 (Teens)'
    elif rating in ['R', 'TV-MA', 'NC-17', 'M', 'MA-17']:
        return 'R/TV-MA (Mature)'
    elif rating in ['NR', 'UR', 'NOT RATED']:
        return 'Not Rated'
    else:
        return 'Other'

def analyze_ratings(all_platforms):
    results = {}
    all_platforms['rating_clean'] = all_platforms['rating'].fillna('Unknown')
    all_platforms['rating_standard'] = all_platforms['rating_clean'].apply(standardize_rating)
    
    # Rating distribution by platform
    rating_by_platform = pd.crosstab(all_platforms['platform'], 
                                     all_platforms['rating_standard'])
    results['rating_counts'] = {
        platform: {rating: int(count) for rating, count in row.items()}
        for platform, row in rating_by_platform.iterrows()
    }
    
    # Rating percentages
    rating_pct = pd.crosstab(all_platforms['platform'], 
                             all_platforms['rating_standard'],
                             normalize='index') * 100
    results['rating_percentages'] = {
        platform: {rating: round(rating_pct, 2) for rating, count in row.items()}
        for platform, row in rating_pct.iterrows()
    }
    
    # Mature content percentage
    mature_ratings = ['TV-MA (Mature)', 'R/TV-MA (Mature)']
    mature_content = all_platforms[all_platforms['rating_standard'].isin(mature_ratings)]
    mature_pct = (mature_content.groupby('platform').size() / 
                  all_platforms.groupby('platform').size() * 100)
    results['mature_content_percentage'] = mature_pct.round(2).to_dict()
    
    # Child-friendly content percentage
    child_ratings = ['TV-Y/TV-Y7 (Children)', 'G (General Audience)', 'TV-G (General Audience)']
    child_content = all_platforms[all_platforms['rating_standard'].isin(child_ratings)]
    child_pct = (child_content.groupby('platform').size() / 
                 all_platforms.groupby('platform').size() * 100)
    results['child_friendly_percentage'] = child_pct.round(2).to_dict()
    
    # Rating distribution by content type
    rating_by_type = pd.crosstab([all_platforms['platform'], all_platforms['type']], 
                                  all_platforms['rating_standard'])
    
    # Convert to nested dictionary
    results['rating_by_content_type'] = {}
    for (platform, content_type), row in rating_by_type.iterrows():
        if platform not in results['rating_by_content_type']:
            results['rating_by_content_type'][platform] = {}
        results['rating_by_content_type'][platform][content_type] = {
            rating: int(count) for rating, count in row.items()
        }
        
    return results

def analyze_unique_content(netflix_df, amazon_df, hulu_df, all_platforms):
    results = {}
    netflix_titles = set(netflix_df['title'])
    amazon_titles = set(amazon_df['title'])
    hulu_titles = set(hulu_df['title'])
    
    # Find duplicated titles across platforms
    title_counts = all_platforms['title'].value_counts()
    shared_titles = title_counts[title_counts > 1].index.tolist()
    
    # Calculate exclusive content (content specific to that platform)
    netflix_only = netflix_titles - (amazon_titles | hulu_titles)
    amazon_only = amazon_titles - (netflix_titles | hulu_titles)
    hulu_only = hulu_titles - (netflix_titles | amazon_titles)
    
    # Calculate overlap between platforms
    netflix_amazon_only = (netflix_titles & amazon_titles) - hulu_titles
    netflix_hulu_only = (netflix_titles & hulu_titles) - amazon_titles
    amazon_hulu_only = (amazon_titles & hulu_titles) - netflix_titles
    all_common = netflix_titles & amazon_titles & hulu_titles
    
    # Store content counts
    results['exclusive_counts'] = {
        'Netflix': len(netflix_only),
        'Amazon Prime': len(amazon_only),
        'Hulu': len(hulu_only)
    }
    
    # Store overlap counts
    results['overlap_counts'] = {
        'Netflix_Amazon': len(netflix_amazon_only),
        'Netflix_Hulu': len(netflix_hulu_only),
        'Amazon_Hulu': len(amazon_hulu_only),
        'All_Platforms': len(all_common)
    }
    
    # Calculate exclusivity percentages
    results['exclusive_percentages'] = {
        'Netflix': round(len(netflix_only) / len(netflix_titles) * 100, 2),
        'Amazon Prime': round(len(amazon_only) / len(amazon_titles) * 100, 2),
        'Hulu': round(len(hulu_only) / len(hulu_titles) * 100, 2)
    }
    
    # Shared content details (up to 10 examples of each)
    results['shared_content_examples'] = {
        'Netflix_Amazon': list(netflix_amazon_only)[:10],
        'Netflix_Hulu': list(netflix_hulu_only)[:10],
        'Amazon_Hulu': list(amazon_hulu_only)[:10],
        'All_Platforms': list(all_common)[:10]
    }
    
    return results

def analyze_countries(all_platforms):
    results = {}
    all_platforms['country_clean'] = all_platforms['country'].fillna('Unknown')
    
    # Extract all countries (multiple countries can be listed)
    all_countries = []
    platform_countries = {platform: [] for platform in all_platforms['platform'].unique()}
    
    for _, row in all_platforms.iterrows():
        if pd.notna(row['country']) and row['country'] != 'Unknown':
            countries = [country.strip() for country in str(row['country']).split(',')]
            all_countries.extend(countries)
            platform_countries[row['platform']].extend(countries)
    
    # Overall country counts
    overall_countries = Counter(all_countries)
    results['top_countries_overall'] = dict(overall_countries.most_common(20))
    
    # Country counts by platform
    results['top_countries_by_platform'] = {}
    for platform, countries in platform_countries.items():
        plt_countries = Counter(countries)
        results['top_countries_by_platform'][platform] = dict(plt_countries.most_common(10))
    
    # US content percentage
    us_content = all_platforms[all_platforms['country_clean'].str.contains('United States', na=False)]
    us_pct = (us_content.groupby('platform').size() / 
              all_platforms.groupby('platform').size() * 100)
    results['us_content_percentage'] = us_pct.round(2).to_dict()
    
    # International content percentage (non-US)
    intl_content = all_platforms[~all_platforms['country_clean'].str.contains('United States', na=False) & 
                               (all_platforms['country_clean'] != 'Unknown')]
    intl_pct = (intl_content.groupby('platform').size() / 
                all_platforms.groupby('platform').size() * 100)
    results['international_content_percentage'] = intl_pct.round(2).to_dict()
    
    # Country diversity (number of unique countries)
    results['country_diversity'] = {platform: len(set(countries)) 
                                    for platform, countries in platform_countries.items()}
    
    # Content by continent
    continent_mapping = {
        'North America': ['United States', 'Canada', 'Mexico'],
        'Europe': ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Russia'],
        'Asia': ['India', 'China', 'Japan', 'South Korea', 'Taiwan'],
        'South America': ['Brazil', 'Argentina', 'Colombia'],
        'Oceania': ['Australia', 'New Zealand'],
        'Africa': ['South Africa', 'Nigeria', 'Kenya', 'Egypt']
    }
    
    # Create reverse mapping (country to continent)
    country_to_continent = {}
    for continent, countries in continent_mapping.items():
        for country in countries:
            country_to_continent[country.lower()] = continent
    
    # Function to assign continent
    def get_continent(country_str):
        if pd.isna(country_str) or country_str == 'Unknown':
            return 'Unknown'
        
        for country in str(country_str).split(','):
            country = country.strip().lower()
            for known_country, continent in country_to_continent.items():
                if known_country in country:
                    return continent
        return 'Other'
    
    all_platforms['continent'] = all_platforms['country_clean'].apply(get_continent)
    
    # Continent distribution
    continent_counts = pd.crosstab(all_platforms['platform'], all_platforms['continent'])
    results['content_by_continent'] = {
        platform: {continent: int(count) for continent, count in row.items()}
        for platform, row in continent_counts.iterrows()
    }
    
    return results

def generate_platform_summary(netflix_df, amazon_df, hulu_df, all_platforms):
    current_year = datetime.now().year
    
    # Calculate content age
    all_platforms['content_age'] = current_year - all_platforms['release_year']
    
    # Summary statistics
    platform_summary = all_platforms.groupby('platform').agg(
        total_titles=('show_id', 'count'),
        movies=('type', lambda x: (x == 'Movie').sum()),
        tv_shows=('type', lambda x: (x == 'TV Show').sum()),
        avg_release_year=('release_year', 'mean'),
        median_release_year=('release_year', 'median'),
        earliest_title=('release_year', 'min'),
        latest_title=('release_year', 'max'),
        avg_content_age=('content_age', 'mean'),
        unique_countries=('country', lambda x: x.nunique()),
        unique_directors=('director', lambda x: x.nunique()),
    )
    
    # Calculate movie percentage
    platform_summary['movie_percentage'] = (platform_summary['movies'] / platform_summary['total_titles'] * 100).round(2)
    
    # Calculate TV show percentage
    platform_summary['tv_show_percentage'] = (platform_summary['tv_shows'] / platform_summary['total_titles'] * 100).round(2)
    
    # Calculate recent content (last 3 years)
    recent_years = list(range(current_year-3, current_year+1))
    recent_content = all_platforms[all_platforms['release_year'].isin(recent_years)]
    recent_counts = recent_content.groupby('platform').size()
    
    platform_summary['recent_content_count'] = recent_counts
    platform_summary['recent_content_percentage'] = (recent_counts / platform_summary['total_titles'] * 100).round(2)
    
    # Calculate mature content percentage
    all_platforms['rating_clean'] = all_platforms['rating'].fillna('Unknown')
    all_platforms['rating_standard'] = all_platforms['rating_clean'].apply(standardize_rating)
    
    mature_ratings = ['TV-MA (Mature)', 'R/TV-MA (Mature)']
    mature_content = all_platforms[all_platforms['rating_standard'].isin(mature_ratings)]
    mature_counts = mature_content.groupby('platform').size()
    
    platform_summary['mature_content_count'] = mature_counts
    platform_summary['mature_content_percentage'] = (mature_counts / platform_summary['total_titles'] * 100).round(2)
    
    platform_summary['avg_release_year'] = platform_summary['avg_release_year'].round(2)
    platform_summary['avg_content_age'] = platform_summary['avg_content_age'].round(2)
    summary_dict = platform_summary.to_dict()
    
    # Content exclusivity metrics
    netflix_titles = set(netflix_df['title'])
    amazon_titles = set(amazon_df['title'])
    hulu_titles = set(hulu_df['title'])
    
    exclusive_counts = {
        'Netflix': len(netflix_titles - (amazon_titles | hulu_titles)),
        'Amazon Prime': len(amazon_titles - (netflix_titles | hulu_titles)),
        'Hulu': len(hulu_titles - (netflix_titles | amazon_titles))
    }
    
    exclusive_percentages = {
        'Netflix': round(exclusive_counts['Netflix'] / len(netflix_titles) * 100, 2),
        'Amazon Prime': round(exclusive_counts['Amazon Prime'] / len(amazon_titles) * 100, 2),
        'Hulu': round(exclusive_counts['Hulu'] / len(hulu_titles) * 100, 2)
    }
    
    summary_dict['exclusive_content_count'] = exclusive_counts
    summary_dict['exclusive_content_percentage'] = exclusive_percentages
    return summary_dict

def run_full_analysis(netflix_path, amazon_path, hulu_path, save_combined=True):
    netflix_df, amazon_df, hulu_df, all_platforms, dataset_info = load_datasets(
        netflix_path, amazon_path, hulu_path)
    
    if save_combined:
        import os
        os.makedirs('Data', exist_ok=True)
        all_platforms.to_csv('Data/all_platforms_combined.csv', index=False)
    
    # Prepare results dictionary
    results = {
        "dataset_info": dataset_info
    } 
    
    # Basic statistics
    results['netflix_stats'] = get_basic_stats(netflix_df, 'Netflix')
    results['amazon_stats'] = get_basic_stats(amazon_df, 'Amazon Prime')
    results['hulu_stats'] = get_basic_stats(hulu_df, 'Hulu')
    
    # Content type analysis
    results['content_type_analysis'] = analyze_content_types(all_platforms)
    
    # Genre analysis
    results['genre_analysis'] = analyze_genres(all_platforms)
    
    # Content age analysis
    results['content_age_analysis'] = analyze_content_age(all_platforms)
    
    # Ratings analysis
    results['ratings_analysis'] = analyze_ratings(all_platforms)
    
    # Unique content analysis
    results['unique_content_analysis'] = analyze_unique_content(
        netflix_df, amazon_df, hulu_df, all_platforms)
    
    
    # Country analysis
    results['country_analysis'] = analyze_countries(all_platforms)
    
    # Platform summary
    results['platform_summary'] = generate_platform_summary(
        netflix_df, amazon_df, hulu_df, all_platforms)
    
    return results

def create_analysis_report(analysis_results):
    report = []
    
    # Add header
    report.append("# Streaming Platforms Content Analysis Report")
    report.append("## Netflix vs. Amazon Prime vs. Hulu")
    report.append("")
    
    # Add dataset information
    dataset_info = analysis_results['dataset_info']
    report.append("## Dataset Information")
    report.append("")
    
    # Before cleaning
    report.append("### Original Datasets")
    report.append("")
    report.append("| Platform | Rows | Columns |")
    report.append("|----------|------|---------|")
    
    for platform, info in dataset_info['before_cleaning'].items():
        report.append(f"| {platform} | {info['rows']} | {info['columns']} |")
    
    report.append("")
    
    # After cleaning
    report.append("### After Cleaning")
    report.append("")
    report.append("| Platform | Rows | Columns | Rows Removed |")
    report.append("|----------|------|---------|--------------|")
    
    for platform in dataset_info['before_cleaning'].keys():
        before = dataset_info['before_cleaning'][platform]['rows']
        after = dataset_info['after_cleaning'][platform]['rows']
        removed = before - after
        removed_pct = round((removed / before * 100), 2) if before > 0 else 0
        
        report.append(f"| {platform} | {after} | {dataset_info['after_cleaning'][platform]['columns']} | {removed} ({removed_pct}%) |")
    
    report.append("")
    
    # Combined dataset
    report.append("### Combined Dataset")
    report.append("")
    report.append(f"Total records: {dataset_info['combined']['rows']}")
    report.append(f"Total columns: {dataset_info['combined']['columns']}")
    report.append("")

    
    # Platform overview
    report.append("## 1. Platform Overview")
    report.append("")
    
    # Summary stats
    summary = analysis_results['platform_summary']
    
    # Overview table
    overview_rows = []
    overview_rows.append("| Metric | Netflix | Amazon Prime | Hulu |")
    overview_rows.append("|--------|---------|--------------|------|")
    
    metrics = [
        ('Total Titles', 'total_titles'),
        ('Movies', 'movies'),
        ('TV Shows', 'tv_shows'),
        ('Movie %', 'movie_percentage'),
        ('TV Show %', 'tv_show_percentage'),
        ('Avg. Release Year', 'avg_release_year'),
        ('Median Release Year', 'median_release_year'),
        ('Avg. Content Age (years)', 'avg_content_age'),
        ('Recent Content %', 'recent_content_percentage'),
        ('Exclusive Content %', 'exclusive_content_percentage')
    ]
    
    for label, key in metrics:
        row = f"| {label} | "
        for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
            value = summary[key][platform]
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
            row += f"{formatted_value} | "
        overview_rows.append(row)
    
    report.extend(overview_rows)
    report.append("")
    
    # Content Type Analysis
    report.append("## 2. Content Type Analysis")
    report.append("")
    report.append("### Movie to TV Show Ratio")
    
    content_types = analysis_results['content_type_analysis']
    
    # Ratio table
    ratio_rows = []
    ratio_rows.append("| Platform | Movies | TV Shows | Movie:TV Ratio |")
    ratio_rows.append("|----------|--------|----------|---------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        movies = content_types['type_counts']['Movie'][platform]
        tv_shows = content_types['type_counts'].get('TV Show', {}).get(platform, 0)
        ratio = content_types['movie_to_tv_ratio'].get(platform, "N/A")
        
        if isinstance(ratio, float):
            ratio_formatted = f"{ratio:.2f}:1"
        else:
            ratio_formatted = "N/A"
            
        ratio_rows.append(f"| {platform} | {movies} | {tv_shows} | {ratio_formatted} |")
    
    report.extend(ratio_rows)
    report.append("")
    
    # Recent Content Analysis
    report.append("### Recent Content (Last 5 Years)")
    report.append("")
    report.append("| Platform | Recent Titles | % of Library |")
    report.append("|----------|---------------|-------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        recent_count = content_types['recent_content_counts'].get(platform, 0)
        recent_pct = content_types['recent_content_percentage'].get(platform, 0)
        report.append(f"| {platform} | {recent_count} | {recent_pct:.2f}% |")
    
    report.append("")
    
    # Genre Analysis
    report.append("## 3. Genre Analysis")
    report.append("")
    
    genres = analysis_results['genre_analysis']
    
    # Top genres by platform
    report.append("### Top 5 Genres by Platform")
    report.append("")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        report.append(f"#### {platform}")
        report.append("")
        report.append("| Genre | Title Count |")
        report.append("|-------|------------|")
        
        platform_genres = genres['top_genres_by_platform'][platform]
        for genre, count in list(platform_genres.items())[:5]:
            report.append(f"| {genre} | {count} |")
        
        report.append("")
    
    # Genre diversity
    report.append("### Genre Diversity")
    report.append("")
    report.append("| Platform | Unique Genres | Exclusive Genres |")
    report.append("|----------|--------------|-----------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        unique_genres = genres['genre_diversity'][platform]
        exclusive_genres = len(genres['unique_genres'][platform])
        report.append(f"| {platform} | {unique_genres} | {exclusive_genres} |")
    
    report.append("")
    
    # Content Age Analysis
    report.append("## 4. Content Age Analysis")
    report.append("")
    
    age_data = analysis_results['content_age_analysis']
    
    # Average content age
    report.append("### Content Age Metrics")
    report.append("")
    report.append("| Platform | Average Age (years) | Median Age (years) | Content Freshness (%) |")
    report.append("|----------|--------------------|-------------------|---------------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        avg_age = age_data['average_content_age'][platform]
        median_age = age_data['median_content_age'][platform]
        freshness = age_data['content_freshness_pct'][platform]
        
        report.append(f"| {platform} | {avg_age:.2f} | {median_age:.1f} | {freshness:.2f} |")
    
    report.append("")
    
    # Age distribution
    report.append("### Content Age Distribution")
    report.append("")
    report.append("| Age Category | Netflix | Amazon Prime | Hulu |")
    report.append("|--------------|---------|--------------|------|")
    
    age_categories = ['0-2 years', '3-5 years', '6-10 years', '11-20 years', '21-50 years', '50+ years']
    
    for category in age_categories:
        row = f"| {category} | "
        for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
            value = age_data['age_categories_pct'][platform].get(category, 0)
            row += f"{value:.2f}% | "
        report.append(row)
    
    report.append("")
    
    # Ratings Analysis
    report.append("## 5. Content Ratings Analysis")
    report.append("")
    
    ratings = analysis_results['ratings_analysis']
    
    # Mature content
    report.append("### Mature vs. Child-Friendly Content")
    report.append("")
    report.append("| Platform | Mature Content (%) | Child-Friendly Content (%) |")
    report.append("|----------|--------------------|-----------------------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        mature = ratings['mature_content_percentage'].get(platform, 0)
        child = ratings['child_friendly_percentage'].get(platform, 0)
        
        report.append(f"| {platform} | {mature:.2f}% | {child:.2f}% |")
    
    report.append("")
    
    # Country Analysis
    report.append("## 6. Content Origin Analysis")
    report.append("")
    
    countries = analysis_results['country_analysis']
    
    # US vs International
    report.append("### US vs. International Content")
    report.append("")
    report.append("| Platform | US Content (%) | International Content (%) | Country Diversity |")
    report.append("|----------|----------------|---------------------------|-------------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        us_pct = countries['us_content_percentage'].get(platform, 0)
        intl_pct = countries['international_content_percentage'].get(platform, 0)
        diversity = countries['country_diversity'].get(platform, 0)
        
        report.append(f"| {platform} | {us_pct:.2f}% | {intl_pct:.2f}% | {diversity} |")
    
    report.append("")
    
    # Top production countries
    report.append("### Top 3 Production Countries by Platform")
    report.append("")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        report.append(f"#### {platform}")
        report.append("")
        report.append("| Country | Title Count |")
        report.append("|---------|------------|")
        
        platform_countries = countries['top_countries_by_platform'][platform]
        for country, count in list(platform_countries.items())[:3]:
            report.append(f"| {country} | {count} |")
        
        report.append("")
    
    # Content Exclusivity
    report.append("## 7. Content Exclusivity Analysis")
    report.append("")
    
    exclusivity = analysis_results['unique_content_analysis']
    
    # Exclusive content stats
    report.append("### Platform Exclusivity")
    report.append("")
    report.append("| Platform | Exclusive Titles | Exclusivity (%) |")
    report.append("|----------|------------------|-----------------|")
    
    for platform in ['Netflix', 'Amazon Prime', 'Hulu']:
        exclusive_count = exclusivity['exclusive_counts'][platform]
        exclusive_pct = exclusivity['exclusive_percentages'][platform]
        
        report.append(f"| {platform} | {exclusive_count} | {exclusive_pct:.2f}% |")
    
    report.append("")
    
    # Content overlap
    report.append("### Content Overlap")
    report.append("")
    report.append("| Platforms | Shared Titles |")
    report.append("|-----------|---------------|")
    report.append(f"| Netflix & Amazon Prime | {exclusivity['overlap_counts']['Netflix_Amazon']} |")
    report.append(f"| Netflix & Hulu | {exclusivity['overlap_counts']['Netflix_Hulu']} |")
    report.append(f"| Amazon Prime & Hulu | {exclusivity['overlap_counts']['Amazon_Hulu']} |")
    report.append(f"| All Three Platforms | {exclusivity['overlap_counts']['All_Platforms']} |")
    
    report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Example usage:
    # Paths to the datasets
    netflix_path = "Data/netflix_titles.csv"
    amazon_path = "Data/amazon_prime_titles.csv"
    hulu_path = "Data/hulu_titles.csv"
    
    # Run analysis
    print("Starting analysis...")
    results = run_full_analysis(netflix_path, amazon_path, hulu_path)
    
    # Generate report
    print("Generating report...")
    report = create_analysis_report(results)
    
    # Save report to file
    with open("Results/streaming_platforms_analysis_report.md", "w") as f:
        f.write(report)
    
    print("Analysis complete. Report saved to streaming_platforms_analysis_report.md")