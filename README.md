# Cross-Platform Movie Recommendation System

This project implements a recommendation system for movies and TV shows across multiple streaming platforms (Netflix, Amazon Prime, and Hulu) using machine learning and content-based filtering techniques.

## Requirements

This project requires Python 3.8+ and the following packages:
```
pandas>=2.2.0
numpy>=1.26.3
matplotlib>=3.8.2
seaborn>=0.13.1
flask>=2.0.0
flask-cors>=3.0.10
scikit-learn>=1.0.0
joblib>=1.1.0
xgboost>=1.5.0
```

## Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data**:
   Make sure your data files are in the correct location:
   - Place streaming platform CSV files in the `Data/` directory
   - Ensure `all_platforms_combined.csv` is generated or available

## Running the Application

### 1. Start the Backend API

First, start the recommendation API server:

```bash
python recommendation_api.py
```

This will start the Flask server on http://localhost:8080. The API provides endpoints for:
- Getting sample titles
- Searching content
- Generating recommendations
- Fetching platform statistics

### 2. View the Web Application

There are two ways to access the web application:

#### Option 1: Using a simple HTTP server
For Python 3:
```bash
python -m http.server 8000
```
Then open http://localhost:8000/recommendation_page.html in your web browser.

#### Option 2: Directly open HTML files
Open `recommendation_page.html` directly in your web browser to see the recommendation interface.

You can also explore other HTML files:
- `data_explanation.html` - Data analysis visualizations
- `feature_analysis.html` - Feature engineering and selection insights
- `model_comparison.html` - Comparison of different recommendation models
- `evaluation.html` - Evaluation metrics and model performance

## Using the Recommendation System

1. **Filtering content**:
   - Select a streaming platform (Netflix, Amazon Prime, Hulu, or All)
   - Choose content type (Movie, TV Show, or All)
   - Click "Apply Filters" to update the displayed titles

2. **Searching**:
   - Use the search bar to find specific titles
   - Results will be filtered based on your platform and type selections

3. **Getting recommendations**:
   - Click on any title card to see XGBoost-powered recommendations
   - Recommendations show similarity scores and content details
   - The system uses a hybrid approach combining machine learning and content-based filtering

## Troubleshooting

- If you see a connection error, ensure the Flask API server is running on port 8080
- Check console logs for detailed error messages
- Make sure all data files are in the correct locations
- Verify that all required Python packages are installed


## Exploring the Source Code

Key files to review:
- `model_training.py` - Machine learning model training pipeline
- `feature_engineering.py` - Feature extraction and processing 
- `feature_selection.py` - Feature selection techniques
- `recommendation_api.py` - Flask API for serving recommendations
- `app.js` - Frontend JavaScript for the web interface

#### Demo Link: https://drive.google.com/drive/folders/1CyCiBmm_0rlp3sI18li1EnGCYSWhkpEw?usp=drive_link
#### PPT: https://drive.google.com/drive/folders/18L7ae-b-ucnAuQA-7nNxa8Vzmp-2WH3J?usp=sharing
