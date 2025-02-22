import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib_venn import venn3, venn2
import matplotlib.gridspec as gridspec

from analysis import (
    load_datasets,
    analyze_content_types,
    analyze_genres,
    analyze_content_age,
    analyze_ratings,
    analyze_unique_content,
    analyze_countries,
    clean_datasets,
    standardize_rating
)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
PLATFORM_COLORS = {
    'Netflix': '#E50914',  # Netflix red
    'Amazon Prime': '#00A8E1',  # Amazon blue
    'Hulu': '#3DBB3D'  # Hulu green
}

def setup_visualization_directory():
    if not os.path.exists('Visualizations'):
        os.makedirs('Visualizations')
    print(f"Visualization directory set up at: {os.path.abspath('Visualizations')}")

def save_figure(plt, filename, dpi=300, bbox_inches='tight'):
    filepath = os.path.join('Visualizations', filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
    print(f"Saved: {filepath}")

def visualize_platform_overview(netflix_df, amazon_df, hulu_df):
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 1. Content count visualization
    ax1 = plt.subplot(gs[0, 0])
    counts = [len(netflix_df), len(amazon_df), len(hulu_df)]
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    colors = [PLATFORM_COLORS[p] for p in platforms]
    
    bars = ax1.bar(platforms, counts, color=colors)
    ax1.set_title('Total Content Library Size', fontsize=14, pad=20)
    ax1.set_ylabel('Number of Titles')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height + 50,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=11
        )
    
    # 2. Movies vs TV Shows - Stacked bar chart
    ax2 = plt.subplot(gs[0, 1])
    movie_counts = []
    tvshow_counts = []
    for df in [netflix_df, amazon_df, hulu_df]:
        movie_counts.append(len(df[df['type'] == 'Movie']))
        tvshow_counts.append(len(df[df['type'] == 'TV Show']))
    
    width = 0.7
    ax2.bar(platforms, movie_counts, width, label='Movies', color=['#f8bbd0', '#bbdefb', '#c8e6c9'])
    ax2.bar(platforms, tvshow_counts, width, bottom=movie_counts, label='TV Shows', color=[PLATFORM_COLORS[p] for p in platforms])
    ax2.set_title('Content Type Distribution', fontsize=14, pad=20)
    ax2.set_ylabel('Number of Titles')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, platform in enumerate(platforms):
        total = movie_counts[i] + tvshow_counts[i]
        movie_pct = movie_counts[i] / total * 100
        tv_pct = tvshow_counts[i] / total * 100
        
        # Movie text
        ax2.text(
            i, movie_counts[i]/2,
            f'{movie_counts[i]:,}\n({movie_pct:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='black'
        )
        # TV Show text
        ax2.text(
            i, movie_counts[i] + tvshow_counts[i]/2,
            f'{tvshow_counts[i]:,}\n({tv_pct:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white'
        )
    plt.suptitle('Platform Content Overview', fontsize=16, y=1.05)
    plt.tight_layout()
    save_figure(plt, 'platform_overview.png')

def visualize_content_type_analysis(all_platforms):
    content_analysis = analyze_content_types(all_platforms)

    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.2, 1])
    
    # 1. Movie to TV Ratio - Horizontal bar chart (Top left)
    ax1 = plt.subplot(gs[0, 0])
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    movie_counts = [content_analysis['type_counts']['Movie'][p] for p in platforms]
    tv_counts = [content_analysis['type_counts'].get('TV Show', {}).get(p, 0) for p in platforms]
    
    totals = [m + t for m, t in zip(movie_counts, tv_counts)]
    movie_pcts = [m / total * 100 for m, total in zip(movie_counts, totals)]

    ax1.barh(platforms, movie_pcts, color=[PLATFORM_COLORS[p] for p in platforms], alpha=0.7)
    ax1.barh(platforms, [100 - m for m in movie_pcts], left=movie_pcts, color='lightgray', alpha=0.5)

    for i, (m_pct, t_pct) in enumerate(zip(movie_pcts, [100 - m for m in movie_pcts])):
        if m_pct > 10:
            ax1.text(m_pct/2, i, f'Movies\n{m_pct:.1f}%', ha='center', va='center', 
                    color='white', fontweight='bold')
        if t_pct > 10:
            ax1.text(m_pct + t_pct/2, i, f'TV Shows\n{t_pct:.1f}%', ha='center', va='center', 
                    color='gray', fontweight='bold')

    ax1.set_title('Movie vs. TV Show Split', fontsize=14, pad=20)
    ax1.set_xlim(0, 120)
    ax1.set_xticks(range(0, 101, 20))
    ax1.set_xlabel('Percentage of Content Library (%)')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, platform in enumerate(platforms):
        ratio = content_analysis['movie_to_tv_ratio'].get(platform, float('inf'))
        if ratio != float('inf'):
            ratio_text = f'Ratio = {ratio:.1f}:1'
        else:
            ratio_text = 'Ratio = ∞:1'
        ax1.text(102, i, ratio_text, va='center', fontsize=9, ha='left')
    
    # 2. Recent content percentage (Top right)
    ax2 = plt.subplot(gs[0, 1])
    recent_pcts = [content_analysis['recent_content_percentage'].get(p, 0) for p in platforms]
    bars = ax2.barh(platforms, recent_pcts, color=[PLATFORM_COLORS[p] for p in platforms], alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=10)
    
    ax2.set_title('Recent Content (2016 - 2021)', fontsize=14, pad=20)
    ax2.set_xlabel('Percentage of Library (%)')
    ax2.set_xlim(0, max(recent_pcts) * 1.2)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Content trends (Bottom spanning both columns)
    ax3 = plt.subplot(gs[1, :])
    current_year = datetime.now().year
    recent_years = list(range(current_year - 10, current_year + 1))

    release_trends = content_analysis['release_trends']
    available_years = sorted([year for year in release_trends.keys() if year in recent_years])

    bar_width = 0.25
    x = np.arange(len(available_years))
    for i, platform in enumerate(platforms):
        counts = [release_trends.get(year, {}).get(platform, 0) for year in available_years]
        bars = ax3.bar(x + (i-1)*bar_width, counts, bar_width, 
                    color=PLATFORM_COLORS[platform], label=platform, alpha=0.8)
        
        # Add numerical values on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label bars with positive values
                ax3.text(bar.get_x() + bar.get_width()/2, height, str(height), 
                        ha='center', va='bottom', fontsize=10)

    ax3.set_title('Content Additions by Release Year', fontsize=14, pad=20)
    ax3.set_xlabel('Release Year')
    ax3.set_ylabel('Number of Titles')
    ax3.set_xticks(x)
    ax3.set_xticklabels(available_years, rotation=45)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Content Type and Release Year Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(plt, 'content_type_analysis.png')

def visualize_genre_analysis(all_platforms):
    genre_analysis = analyze_genres(all_platforms)
    plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])
    
    # 1. Top genres overall - Horizontal bar chart (Top left)
    ax1 = plt.subplot(gs[0, 0])
    top_genres = list(genre_analysis['top_genres_overall'].items())[:10]
    genres, counts = zip(*top_genres)

    bars = ax1.barh(list(reversed(genres)), list(reversed(counts)), color='#6A5ACD', alpha=0.7)
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 10, bar.get_y() + bar.get_height()/2, f'{int(width):,}', 
                va='center', fontsize=9)

    ax1.set_title('Top 10 Genres Across All Platforms', fontsize=14, pad=20)
    ax1.set_xlabel('Number of Titles')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Top genres by platform - Grouped bar chart (Top right)
    platform_top_genres = {}
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    for platform in platforms:
        platform_top_genres[platform] = dict(list(genre_analysis['top_genres_by_platform'][platform].items())[:5])
    
    # 3. Genre diversity - Bar chart (Bottom left)
    ax3 = plt.subplot(gs[1, 0])
    diversity = [genre_analysis['genre_diversity'][p] for p in platforms]
    unique_genres = [len(genre_analysis['unique_genres'][p]) for p in platforms]
    
    x = np.arange(len(platforms))
    width = 0.35
    bars1 = ax3.bar(x - width/2, diversity, width, 
                    color=[PLATFORM_COLORS[p] for p in platforms], 
                    alpha=0.8, label='Total Unique Genres')
    bars2 = ax3.bar(x + width/2, unique_genres, width, 
                    color=[PLATFORM_COLORS[p] for p in platforms], 
                    alpha=0.4, label='Platform-Exclusive Genres')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                    str(int(height)), ha='center', va='bottom', fontsize=10)

    ax3.set_title('Genre Diversity by Platform', fontsize=14, pad=20)
    ax3.set_ylabel('Number of Genres')
    ax3.set_xticks(x)
    ax3.set_xticklabels(platforms)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Genre Analysis Across Streaming Platforms', fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(plt, 'genre_analysis.png')

def visualize_content_age(all_platforms):
    age_analysis = analyze_content_age(all_platforms)
    plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 1. Average and median content age - Bar chart
    ax1 = plt.subplot(gs[0, 0])
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    avg_ages = [age_analysis['average_content_age'][p] for p in platforms]
    median_ages = [age_analysis['median_content_age'][p] for p in platforms]

    x = np.arange(len(platforms))
    width = 0.35
    bars1 = ax1.bar(x - width/2, avg_ages, width, label='Average Age', color='#2E86C1', alpha=0.8)
    bars2 = ax1.bar(x + width/2, median_ages, width, label='Median Age', color='#3498DB', alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3, 
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Average and Median Content Age by Platform', fontsize=14, pad=20)
    ax1.set_ylabel('Age in Years')
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(avg_ages) * 1.2)
    
    # 2. Content freshness - Donut charts
    ax2 = plt.subplot(gs[0, 1])
    centers = [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]
    
    for i, platform in enumerate(platforms):
        freshness = age_analysis['content_freshness_pct'][platform]
        older = 100 - freshness
        x, y = centers[i]
        wedges, texts = ax2.pie([freshness, older],
                colors=[PLATFORM_COLORS[platform], '#E0E0E0'],
                wedgeprops=dict(width=0.3, edgecolor='w'),
                center=(x, y),
                radius=0.2,
                startangle=90)

        ax2.text(x, y, f"{freshness:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        ax2.text(x, y+0.25, platform, ha='center', va='center', fontsize=10)
        ax2.text(x, y-0.25, "Content < 5 years old", ha='center', va='center', fontsize=8)

    ax2.set_title('Content Freshness', fontsize=14, pad=20)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Age distribution - Grouped bar chart
    ax3 = plt.subplot(gs[1, :])
    age_categories = ['3-5 years', '6-10 years', '11-20 years', '21-50 years', '50+ years']
    x = np.arange(len(age_categories))
    width = 0.25

    # Create bars for each platform
    for i, platform in enumerate(platforms):
        values = [age_analysis['age_categories_pct'][platform].get(category, 0) 
                 for category in age_categories]
        bars = ax3.bar(x + (i-1)*width, values, width, 
                      label=platform, color=PLATFORM_COLORS[platform], alpha=0.8)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            if height >= 2:  # Only show labels for values >= 2%
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=9, rotation=0)

    ax3.set_title('Content Age Distribution by Platform', fontsize=14, pad=20)
    ax3.set_ylabel('Percentage of Content Library (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(age_categories, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max([max(age_analysis['age_categories_pct'][p].values()) 
                        for p in platforms]) * 1.15)

    plt.suptitle('Content Age Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(plt, 'content_age_analysis.png')

def visualize_ratings_analysis(all_platforms):
    ratings_analysis = analyze_ratings(all_platforms)
    plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 1. Overall rating distribution - Stacked bar chart
    ax1 = plt.subplot(gs[0, 0])
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    all_ratings = set()
    for platform in platforms:
        all_ratings.update(ratings_analysis['rating_counts'][platform].keys())

    rating_groups = {
        'Mature': ['R/TV-MA (Mature)', 'TV-MA (Mature)'],
        'Teen': ['PG-13/TV-14 (Teens)', 'TV-14 (14+)'],
        'Family': ['PG (Parental Guidance)', 'TV-PG (Parental Guidance)'],
        'Children': ['G (General Audience)', 'TV-G (General Audience)', 'TV-Y/TV-Y7 (Children)'],
        'Not Rated': ['Not Rated', 'Other', 'Unknown']
    }

    grouped_counts = {platform: {} for platform in platforms}
    for platform in platforms:
        for group, ratings in rating_groups.items():
            count = sum(ratings_analysis['rating_counts'][platform].get(rating, 0) for rating in ratings)
            grouped_counts[platform][group] = count

    x = np.arange(len(platforms))
    bottom = np.zeros(len(platforms))
    group_colors = {
        'Mature': '#D32F2F',     
        'Teen': '#F57C00',      
        'Family': '#FFC107',     
        'Children': '#4CAF50',  
        'Not Rated': '#9E9E9E'   
    }

    for group in ['Children', 'Family', 'Teen', 'Mature', 'Not Rated']:
        values = [grouped_counts[platform].get(group, 0) for platform in platforms]
        ax1.bar(x, values, bottom=bottom, label=group, color=group_colors[group], width=0.6)
        for i, value in enumerate(values):
            total = sum(grouped_counts[platforms[i]].values())
            percentage = (value / total * 100) if total > 0 else 0
            if percentage >= 5: 
                ax1.text(i, bottom[i] + value/2, f'{percentage:.1f}%', 
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if group in ['Mature', 'Not Rated'] else 'black')
        bottom += values

    ax1.set_title('Content Rating Distribution by Platform', fontsize=14)
    ax1.set_ylabel('Number of Titles')
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms)
    ax1.legend(title='Rating Categories')
    ax1.grid(axis='y', alpha=0.3)
    for i, platform in enumerate(platforms):
        total = sum(grouped_counts[platform].values())
        ax1.text(i, total + 5, f'Total: {total:,}', ha='center', va='bottom')
    
    # 2. Mature vs Child-friendly content - Horizontal bar chart
    ax2 = plt.subplot(gs[0, 1])
    mature_pct = [ratings_analysis['mature_content_percentage'].get(p, 0) for p in platforms]
    child_pct = [ratings_analysis['child_friendly_percentage'].get(p, 0) for p in platforms]
    y_pos = np.arange(len(platforms))

    mature_bars = ax2.barh(y_pos - 0.2, mature_pct, height=0.4, color='#D32F2F', label='Mature Content (R/TV-MA)')
    child_bars = ax2.barh(y_pos + 0.2, child_pct, height=0.4, color='#4CAF50', label='Child-Friendly Content (G/TV-Y/TV-G)')
    for bars, offset in [(mature_bars, -0.2), (child_bars, 0.2)]:
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   va='center', ha='left', fontsize=9)
    
    ax2.set_title('Mature vs Child-Friendly Content', fontsize=14)
    ax2.set_xlabel('Percentage of Library (%)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(platforms)
    ax2.set_xlim(0, max(max(mature_pct), max(child_pct)) * 1.2)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    plt.subplots_adjust(top=0.9)
    plt.suptitle('Content Ratings Analysis', fontsize=20, y=0.98)
    save_figure(plt, 'ratings_analysis.png')

def visualize_unique_content(netflix_df, amazon_df, hulu_df, all_platforms):
    unique_analysis = analyze_unique_content(netflix_df, amazon_df, hulu_df, all_platforms)
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
    
    # 1. Content exclusivity - Pie charts
    ax1 = plt.subplot(gs[0, 0])
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    centers = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
    
    for i, platform in enumerate(platforms):
        exclusive_pct = unique_analysis['exclusive_percentages'][platform]
        shared_pct = 100 - exclusive_pct
        x, y = centers[i]
        wedges, texts = ax1.pie([exclusive_pct, shared_pct],
                                colors=[PLATFORM_COLORS[platform], '#E0E0E0'],
                                wedgeprops=dict(width=0.3, edgecolor='w'),
                                center=(x, y),
                                radius=0.2,
                                startangle=90)

        ax1.text(x, y, f"{exclusive_pct:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        ax1.text(x, y+0.25, platform, ha='center', va='center', fontsize=10)
        ax1.text(x, y-0.25, "Exclusive Content", ha='center', va='center', fontsize=8)
    
    ax1.set_title('Content Exclusivity by Platform', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    exclusive_patch = mpatches.Patch(color='gray', label='Exclusive Content')
    shared_patch = mpatches.Patch(color='#E0E0E0', label='Shared Content')
    ax1.legend(handles=[exclusive_patch, shared_patch], loc='upper right')
    
    # 2. Content overlap - Venn diagram
    ax2 = plt.subplot(gs[0, 1])
    netflix_count = len(netflix_df)
    amazon_count = len(amazon_df)
    hulu_count = len(hulu_df)
    
    netflix_amazon = unique_analysis['overlap_counts']['Netflix_Amazon']
    netflix_hulu = unique_analysis['overlap_counts']['Netflix_Hulu']
    amazon_hulu = unique_analysis['overlap_counts']['Amazon_Hulu']
    all_common = unique_analysis['overlap_counts']['All_Platforms']

    netflix_only = unique_analysis['exclusive_counts']['Netflix']
    amazon_only = unique_analysis['exclusive_counts']['Amazon Prime']
    hulu_only = unique_analysis['exclusive_counts']['Hulu']
    
    venn_sizes = (
        netflix_only,              
        amazon_only,              
        netflix_amazon - all_common, 
        hulu_only,              
        netflix_hulu - all_common, 
        amazon_hulu - all_common,   
        all_common               
    )
    
    venn_sizes = tuple(max(0, size) for size in venn_sizes)
    v = venn3(subsets=venn_sizes,
              set_labels=('Netflix', 'Amazon Prime', 'Hulu'))
    
    for i, label in enumerate(v.subset_labels):
        if label and venn_sizes[i] > 0:
            v.subset_labels[i].set_text(f'{venn_sizes[i]:,}')

    patch_ids = ['100', '010', '110', '001', '101', '011', '111']
    for i, patch in enumerate(v.patches):
        if patch and i < len(patch_ids):
            patch_id = patch_ids[i]
            if patch_id == '100':  # Netflix only
                patch.set_color(PLATFORM_COLORS['Netflix'])
            elif patch_id == '010':  # Amazon only
                patch.set_color(PLATFORM_COLORS['Amazon Prime'])
            elif patch_id == '001':  # Hulu only
                patch.set_color(PLATFORM_COLORS['Hulu'])
            else:
                patch.set_color('#9C9C9C')
            patch.set_alpha(0.7)
    
    ax2.set_title('Content Overlap Between Platforms', fontsize=14)
    explanation_text = (
        f"Netflix: {netflix_count:,} titles\n"
        f"Amazon Prime: {amazon_count:,} titles\n"
        f"Hulu: {hulu_count:,} titles\n\n"
        f"Shared on all platforms: {all_common:,} titles"
    )

    ax2.text(0.95, 0.05, explanation_text, 
            transform=ax2.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 3. Platform-pair overlap - Bar chart
    ax4 = plt.subplot(gs[1, :])
    pairs = ['Netflix & Amazon', 'Netflix & Hulu', 'Amazon & Hulu', 'All Three']
    pair_counts = [
        unique_analysis['overlap_counts']['Netflix_Amazon'],
        unique_analysis['overlap_counts']['Netflix_Hulu'],
        unique_analysis['overlap_counts']['Amazon_Hulu'],
        unique_analysis['overlap_counts']['All_Platforms']
    ]

    bars = ax4.bar(pairs, pair_counts, color=['#9575CD', '#7986CB', '#64B5F6', '#4FC3F7'])
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height):,}', 
               ha='center', va='bottom', fontsize=9)
    
    ax4.set_title('Content Overlap Between Platform Pairs', fontsize=14)
    ax4.set_ylabel('Number of Shared Titles')
    ax4.set_xticklabels(pairs, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Content Exclusivity and Overlap Analysis', fontsize=20, y=0.98)
    save_figure(plt, 'content_exclusivity_analysis.png')

def visualize_countries_analysis(all_platforms):
    countries_analysis = analyze_countries(all_platforms)
    plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.8])
    
    # 1. US vs International content - Stacked bar chart
    ax1 = plt.subplot(gs[0, 0])
    platforms = ['Netflix', 'Amazon Prime', 'Hulu']
    us_pct = [countries_analysis['us_content_percentage'].get(p, 0) for p in platforms]
    intl_pct = [countries_analysis['international_content_percentage'].get(p, 0) for p in platforms]
    unknown_pct = [100 - (us + intl) for us, intl in zip(us_pct, intl_pct)]

    ax1.bar(platforms, us_pct, label='US Content', color='#3B5998')
    ax1.bar(platforms, intl_pct, bottom=us_pct, label='International Content', color='#55ACEE')

    for i, platform in enumerate(platforms):
        if us_pct[i] > 5:
            ax1.text(i, us_pct[i]/2, f'{us_pct[i]:.1f}%', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=9)
        if intl_pct[i] > 5:
            ax1.text(i, us_pct[i] + intl_pct[i]/2, f'{intl_pct[i]:.1f}%', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=9)
  
    ax1.set_title('US vs International Content', fontsize=14)
    ax1.set_ylabel('Percentage of Library (%)')
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Country diversity - Bar chart
    ax2 = plt.subplot(gs[0, 1])
    diversity = [countries_analysis['country_diversity'][p] for p in platforms]

    bars = ax2.bar(platforms, diversity, color=[PLATFORM_COLORS[p] for p in platforms])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2, str(height), 
               ha='center', va='bottom', fontsize=10)

    ax2.set_title('Country Diversity (Unique Countries Represented)', fontsize=14)
    ax2.set_ylabel('Number of Countries')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Top production countries - Horizontal bar chart
    ax3 = plt.subplot(gs[1, :])
    all_countries = {}
    for country, count in countries_analysis['top_countries_overall'].items():
        all_countries[country] = count

    sorted_countries = sorted(all_countries.items(), key=lambda x: x[1], reverse=True)[:10]
    countries, _ = zip(*sorted_countries)

    platform_country_data = {}
    for platform in platforms:
        platform_country_data[platform] = []
        for country in countries:
            count = countries_analysis['top_countries_by_platform'].get(platform, {}).get(country, 0)
            platform_country_data[platform].append(count)

    y_pos = np.arange(len(countries))
    left = np.zeros(len(countries))
    for platform in platforms:
        ax3.barh(y_pos, platform_country_data[platform], left=left, 
               color=PLATFORM_COLORS[platform], label=platform)
        for i, value in enumerate(platform_country_data[platform]):
            left[i] += value

    ax3.set_title('Top 10 Production Countries Across Platforms', fontsize=14)
    ax3.set_xlabel('Number of Titles')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([country for country in countries])
    ax3.legend(loc='upper right')
    ax3.grid(axis='x', alpha=0.3)

    for i, country in enumerate(countries):
        total = sum(platform_country_data[platform][i] for platform in platforms)
        ax3.text(total + 5, i, f'Total: {total}', va='center', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.suptitle('Content Origin Analysis', fontsize=20, y=0.98)
    save_figure(plt, 'country_analysis.png')

def create_dashboard(netflix_df, amazon_df, hulu_df, all_platforms):
    setup_visualization_directory()
    visualize_platform_overview(netflix_df, amazon_df, hulu_df)
    visualize_content_type_analysis(all_platforms)
    visualize_genre_analysis(all_platforms)
    visualize_content_age(all_platforms)
    visualize_ratings_analysis(all_platforms)
    visualize_unique_content(netflix_df, amazon_df, hulu_df, all_platforms)
    visualize_countries_analysis(all_platforms)

if __name__ == "__main__":
    netflix_path = "Data/netflix_titles.csv"
    amazon_path = "Data/amazon_prime_titles.csv"
    hulu_path = "Data/hulu_titles.csv"

    netflix_df, amazon_df, hulu_df, all_platforms, _ = load_datasets(netflix_path, amazon_path, hulu_path)
    create_dashboard(netflix_df, amazon_df, hulu_df, all_platforms)