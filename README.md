# Cross Platform Movie Recommendation System

This project aims to develop a cross-platform recommendation application that helps users discover new movies
and TV shows across different streaming services. If a user enjoys a particular show or movie on one platform,
the app will suggest similar content available on another platform. To make this possible, the application will
incorporate a recommendation model that analyzes user preferences and content similarities
Alongside the recommendation engine, the project will feature a simple, user-friendly interface that allows users
to input their favorite content, select their preferred streaming services, and receive personalized
recommendations. The goal is to make the app visually appealing, easy to navigate, and intuitive for all users.

## Requirements

The project requires Python 3.8+ and the following packages:
```
pandas>=2.2.0
numpy>=1.26.3
matplotlib>=3.8.2
seaborn>=0.13.1
jupyter>=1.0.0
ipykernel>=6.29.0
openpyxl>=3.1.2
plotly>=5.18.0
matplotlib-venn
```

## Project Structure

```
cap5771sp25-project/
│
├── Data/                          
│   ├── netflix_titles.csv        
│   ├── amazon_prime_titles.csv  
│   ├── hulu_titles.csv        
│   └── all_platforms_combined.csv 
├── Report/                  
│   └── Milestone1.pdf
├── Results/                  
│   └── streaming_platforms_analysis_report.md
├── Visualizations/          
│   ├── platform_overview.png
│   ├── content_type_analysis.png
│   ├── genre_analysis.png
│   ├── content_age_analysis.png
│   ├── ratings_analysis.png
│   ├── content_exclusivity_analysis.png
│   └── country_analysis.png
|── Scripts/                    
│   ├── analysis.py
│   ├── visualizations.py
│   ├── requirements.txt
│   ├── main.py
```

## Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r Scripts/requirements.txt
```
4. Run main.py:
```bash
python Scripts/main.py
```
