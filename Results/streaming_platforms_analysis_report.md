# Streaming Platforms Content Analysis Report
## Netflix vs. Amazon Prime vs. Hulu

## Dataset Information

### Original Datasets

| Platform | Rows | Columns |
|----------|------|---------|
| Netflix | 8807 | 12 |
| Amazon Prime | 9668 | 12 |
| Hulu | 3073 | 12 |

### After Cleaning

| Platform | Rows | Columns | Rows Removed |
|----------|------|---------|--------------|
| Netflix | 8807 | 15 | 0 (0.0%) |
| Amazon Prime | 9668 | 15 | 0 (0.0%) |
| Hulu | 3073 | 15 | 0 (0.0%) |

### Combined Dataset

Total records: 21548
Total columns: 16

## 1. Platform Overview

| Metric | Netflix | Amazon Prime | Hulu |
|--------|---------|--------------|------|
| Total Titles | 8807 | 9668 | 3073 | 
| Movies | 6131 | 7814 | 1484 | 
| TV Shows | 2676 | 1854 | 1589 | 
| Movie % | 69.62 | 80.82 | 48.29 | 
| TV Show % | 30.38 | 19.18 | 51.71 | 
| Avg. Release Year | 2014.18 | 2008.34 | 2012.57 | 
| Median Release Year | 2017.00 | 2016.00 | 2016.00 | 
| Avg. Content Age (years) | 10.82 | 16.66 | 12.43 | 
| Recent Content % | nan | nan | nan | 
| Exclusive Content % | 93.44 | 94.12 | 86.66 | 

## 2. Content Type Analysis

### Movie to TV Show Ratio
| Platform | Movies | TV Shows | Movie:TV Ratio |
|----------|--------|----------|---------------|
| Netflix | 6131 | 2676 | 2.29:1 |
| Amazon Prime | 7814 | 1854 | 4.21:1 |
| Hulu | 1484 | 1589 | 0.93:1 |

### Recent Content (Last 5 Years)

| Platform | Recent Titles | % of Library |
|----------|---------------|-------------|
| Netflix | 1545 | 17.54% |
| Amazon Prime | 2404 | 24.87% |
| Hulu | 603 | 19.62% |

## 3. Genre Analysis

### Top 5 Genres by Platform

#### Netflix

| Genre | Title Count |
|-------|------------|
| International Movies | 2752 |
| Drama | 2427 |
| Comedy | 1674 |
| International TV Shows | 1351 |
| Documentary | 869 |

#### Amazon Prime

| Genre | Title Count |
|-------|------------|
| Drama | 3687 |
| Comedy | 2099 |
| Action | 1657 |
| Suspense | 1501 |
| Children's's | 1085 |

#### Hulu

| Genre | Title Count |
|-------|------------|
| Drama | 907 |
| Comedy | 667 |
| Adventure | 556 |
| Action | 555 |
| Documentary | 524 |

### Genre Diversity

| Platform | Unique Genres | Exclusive Genres |
|----------|--------------|-----------------|
| Netflix | 42 | 37 |
| Amazon Prime | 31 | 18 |
| Hulu | 36 | 24 |

## 4. Content Age Analysis

### Content Age Metrics

| Platform | Average Age (years) | Median Age (years) | Content Freshness (%) |
|----------|--------------------|-------------------|---------------------|
| Netflix | 10.82 | 8.0 | 17.54 |
| Amazon Prime | 16.66 | 9.0 | 24.87 |
| Hulu | 12.43 | 9.0 | 19.62 |

### Content Age Distribution

| Age Category | Netflix | Amazon Prime | Hulu |
|--------------|---------|--------------|------|
| 0-2 years | 0.00% | 0.00% | 0.00% | 
| 3-5 years | 17.54% | 24.87% | 19.62% | 
| 6-10 years | 53.04% | 31.16% | 40.32% | 
| 11-20 years | 20.53% | 21.89% | 26.13% | 
| 21-50 years | 7.97% | 14.55% | 11.91% | 
| 50+ years | 0.92% | 7.53% | 2.02% | 

## 5. Content Ratings Analysis

### Mature vs. Child-Friendly Content

| Platform | Mature Content (%) | Child-Friendly Content (%) |
|----------|--------------------|-----------------------------|
| Netflix | 45.52% | 10.31% |
| Amazon Prime | 24.16% | 20.08% |
| Hulu | 23.95% | 7.91% |

## 6. Content Origin Analysis

### US vs. International Content

| Platform | US Content (%) | International Content (%) | Country Diversity |
|----------|----------------|---------------------------|-------------------|
| Netflix | 43.53% | 56.47% | 89 |
| Amazon Prime | 3.52% | 96.48% | 26 |
| Hulu | 36.61% | 63.39% | 32 |

### Top 3 Production Countries by Platform

#### Netflix

| Country | Title Count |
|---------|------------|
| United States | 3834 |
| United Kingdom | 1460 |
| India | 995 |

#### Amazon Prime

| Country | Title Count |
|---------|------------|
| United Kingdom | 9031 |
| United States | 340 |
| India | 233 |

#### Hulu

| Country | Title Count |
|---------|------------|
| United Kingdom | 1578 |
| United States | 1125 |
| Japan | 270 |

## 7. Content Exclusivity Analysis

### Platform Exclusivity

| Platform | Exclusive Titles | Exclusivity (%) |
|----------|------------------|-----------------|
| Netflix | 8229 | 93.44% |
| Amazon Prime | 9100 | 94.12% |
| Hulu | 2663 | 86.66% |

### Content Overlap

| Platforms | Shared Titles |
|-----------|---------------|
| Netflix & Amazon Prime | 347 |
| Netflix & Hulu | 189 |
| Amazon Prime & Hulu | 179 |
| All Three Platforms | 42 |
