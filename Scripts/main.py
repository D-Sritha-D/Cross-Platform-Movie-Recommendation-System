import os
import json
from datetime import datetime

from feature_engineering import run_feature_engineering_pipeline
from ContentStreamingRecommendationModel import ContentStreamingRecommendationModel, demo_recommendations

from analysis import (
    load_datasets,
    run_full_analysis,
    create_analysis_report
)
from visualizations import create_dashboard

def setup_directories():
    directories = ['Data', 'Visualizations', 'Results', 'Models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def run_streaming_analysis(netflix_path, amazon_path, hulu_path, output_dir="Results", save_combined=True):
    print(f"\n{'='*50}")
    print(f"STREAMING PLATFORMS ANALYSIS")
    print(f"{'='*50}\n")
    setup_directories()
    
    print("---------------------MileStone 1---------------------")
    print("Step 1: Loading and cleaning datasets...")
    netflix_df, amazon_df, hulu_df, all_platforms, _ = load_datasets(
        netflix_path, amazon_path, hulu_path
    )
    print(f"Loaded datasets successfully:")
    print(f"  - Netflix: {len(netflix_df)} titles")
    print(f"  - Amazon Prime: {len(amazon_df)} titles")
    print(f"  - Hulu: {len(hulu_df)} titles")
    print(f"  - Combined: {len(all_platforms)} titles")
    
    print("\nStep 2: Running full analysis...")
    analysis_results = run_full_analysis(netflix_path, amazon_path, hulu_path, save_combined)
    
    print("\nStep 3: Generating analysis report...")
    report = create_analysis_report(analysis_results)
    report_file = os.path.join(output_dir, "streaming_platforms_analysis_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Analysis report saved to: {report_file}")
    
    print("\nStep 4: Creating visualizations...")
    create_dashboard(netflix_df, amazon_df, hulu_df, all_platforms)
    print(f"Visualizations saved to: Visualizations/")
    print(f"\n{'='*50}")
    print(f"MILESTONE 1 COMPLETE")
    print(f"{'='*50}")

    print("\n---------------------Milestone 2---------------------")
    print("Step 1: Feature Engineering...")
    results = run_feature_engineering_pipeline()
    print(f"Feature engineering complete. Selected {len(results['selected_features'])} features.")
    print(f"Data saved to Data/engineered_features.csv")
    
    print("\nStep 2: Training and evaluating recommendation model...")
    run_recommendation_model()
    
    print(f"\n{'='*50}")
    print(f"MILESTONE 2 COMPLETE")
    print(f"{'='*50}")

def run_recommendation_model():
    # Check if we already have a saved model
    model_path = "Models/cross_platform_recommender.pkl"
    
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}. Running demo...")
        demo_recommendations(model_path)
    else:
        print("No existing model found. Training a new model...")
        if os.path.exists("Data/engineered_features.csv"):
            model = ContentStreamingRecommendationModel()
            model.load_data("Data/engineered_features.csv")
            
            # Save the model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
            
            # Run the demo
            print("\nRunning model demo...")
            demo_recommendations(model_path)
            
            # Compare machine learning models
            print("\nComparing ML models for platform prediction...")
            comparison_results = model.train_and_compare_models()
            
            # Save comparison results
            comparison_file = "Results/ml_model_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"Model comparison results saved to {comparison_file}")
        else:
            print("Error: engineered_features.csv not found. Please run feature engineering first.")

def main():
    netflix_path = "Data/netflix_titles.csv"
    amazon_path = "Data/amazon_prime_titles.csv"
    hulu_path = "Data/hulu_titles.csv"
    output_dir = "Results"
    save_combined = True

    for filepath in [netflix_path, amazon_path, hulu_path]:
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            return 1
        
    run_streaming_analysis(
        netflix_path,
        amazon_path,
        hulu_path,
        output_dir,
        save_combined
    )
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)