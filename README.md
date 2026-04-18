# ♻️ NYC Recycling Performance Prediction
--------
![download](https://github.com/user-attachments/assets/f3573626-eff2-4b12-b6a5-814150b3ef38)

## 1. Project Overview

This project develops a predictive analytics solution to help the **NYC Department of Sanitation(DSNY)** identify community districts at risk of falling below the 20% recycling ratio target. By analyzing historical waste collection patterns, the system forecasts which districts are likely to achieve high recycling performance (>20% recycling ratio) one month in advance.

**Teaser**: This project delivers a deployed predictive model accessible through an interactive Streamlit application that forecasts community districts’ recycling performance one month in advance, correctly identifying 90% of districts at risk of missing the 20% recycling target.

<img width="2940" height="1654" alt="Screenshot 2025-12-11 at 8 11 40 AM" src="https://github.com/user-attachments/assets/3a149337-9281-43ec-a73f-238486d911f5" />

*screenshot of the app interface*

-------
## 2. Primary Business Objectives
1. Predictive Monitoring: Forecast recycling performance at the community district level

2. Resource Optimization: Enable DSNY to proactively allocate education and outreach resources

3. Target Achievement: Support NYC's goal of increasing recycling rates across all boroughs

4. Anomaly Detection: Identify districts with unusually high refuse generation

##### SLIDES PRESENTATION LINK: [CLICK HERE](https://docs.google.com/presentation/d/1RNbVSRKBp2s9ffdBxp6cl1iulaJCS3t2tGzqoKuc85o/edit?usp=sharing)
-----------
## 3. Dataset Information
### 3.1.  Primary Dataset: DSNY Monthly Tonnage Data
Source: NYC Open Data - [DSNY Monthly Tonnage](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5/about_data)
- Coverage: Monthly waste collection data by community district (2022-2025)
- Records: ~2,700 monthly district-level observations
### 3.2. Secondary Dataset: Population Data
- Source: NYC Open Data - [Population by Community District](https://data.cityofnewyork.us/City-Government/New-York-City-Population-By-Community-Districts/xi7c-iiu2/about_data)
- Coverage: 2010 Census data mapped to community districts

**Data dictionary**
| Column                | Description                                                   | Type              |
|-----------------------|---------------------------------------------------------------|-------------------|
| month                 | Reporting month (YYYY-MM)                                     | Period            |
| borough               | NYC borough                                                   | Categorical       |
| communitydistrict     | Community district number (1–18)                              | Integer           |
| refusetonscollected   | Total refuse (non-recyclable) tons                            | Continuous        |
| papertonscollected    | Paper recyclables collected (tons)                            | Continuous        |
| mgptonscollected      | Metal/Glass/Plastic recyclables (tons)                        | Continuous        |
| resorganicstons       | Residential organics (tons)                                   | Continuous        |
| schoolorganictons     | School organics (tons)                                        | Continuous        |
| otherorganicstons     | Other organics (tons)                                         | Continuous        |
| xmastreetons          | Christmas tree collection (tons)                              | Continuous        |
| population_2010       | 2010 Census population                                        | Integer           |
| recycling_ratio       | (Paper + MGP) / (Paper + MGP + Refuse) -engineered                      | Continuous [0–1]  |
| high_recycling        | Binary flag: recycling_ratio > 0.2 -engineered                         | Binary            |
---------
## 4. Environment Setup & Reproduction
### 4.1. Environment setup
```bash
# Clone the repo
git clone [https://github.com/yourusername/project.git](https://github.com/yourusername/project.git)
cd project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 4.2. Data Download
- Download the CSVs from [NYC open data link1](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5/about_data) AND [NYC open data link2](https://data.cityofnewyork.us/City-Government/New-York-City-Population-By-Community-Districts/xi7c-iiu2/about_data)

- Save them to the datasets/ folder (ensure to change the paths accordingly in the notebooks).
- Run eda_clean.ipynb to clean the data and generate data needed for the modeling process.
- Run clean_models_notebook to train and evaluate models, saving files .pkl and config.json
- Run the app: streamlit run app/app.py

**IMPORTANT:** Ensure to check files paths in the loading process.

## 5. Exploratory data analysis
- Our **Eda** focused on understanding **waste composition**, **trends** and **correlations**
For our distribution analysis, histograms were created for: **Refuse tons collected** ,**Paper tons collected**, **MGP tons collected** revealing a strong right skew and a large inter-district variability, justifying standardization of features and month_district level aggregations to conserve district differences.

<img width="1489" height="490" alt="waste_type_distribution" src="https://github.com/user-attachments/assets/d73edcd6-78cb-4600-a5ff-1770be8dc74a" />

- **Pearson correlations**  between refuse, paper, and MGP showed:
- Moderate correlations
- Clear separation between refuse and recyclable streams
This supported modeling them as distinct behavioral signals, not redundant features; (avoiding multicollinearity)

<img width="567" height="503" alt="corelation" src="https://github.com/user-attachments/assets/c1a86b4d-a73f-4038-9b00-34e246e6769a" />

Time Series Analysis for **refuse tons collected** with **Monthly borough-level aggregation** showed:

- Strong seasonality with similar patterns for all borough
- Borough-specific tonnage amount
- Long-term stability suitable for lag-based forecasting

<img width="989" height="490" alt="refuse_borough_over_time" src="https://github.com/user-attachments/assets/c736b979-42d9-4427-b62c-0f912a474886" />

## 6. Feature engineering: Lag Features (Core Predictive Signals)
Lag features were engineered per borough and community district to ensure locality consistency.
### Lag-1 (previous month)
- refuse_lag1
- paper_lag1
- mgp_lag1
### Lag-12 (same month previous year)
- refuse_lag12

These features capture momentum and seasonality, avoid data leakage and reflect what DSNY would realistically know at prediction time\
Missing lag values were filled using group-level historical means, preserving temporal integrity.
### Population Normalization
Population data was merged at the borough–district level.
Derived feature:

```ini
refuse_per_capita_2010 = refusetonscollected / population_2010
```
This allows a **fair comparison** across districts and a **density-adjusted** interpretation.

### Target Engineering
**Recycling Ratio:** The core performance metric:
```ini
recycling_ratio =
(papertonscollected + mgptonscollected) /
(papertonscollected + mgptonscollected + refusetonscollected)
```
This aligns with real-world recycling performance definitions.

**Classification Target:** high_recycling

Binary target defined as:
```ini
high_recycling = 1 if recycling_ratio > 0.20 else 0
```
**Rationale:**

- 20% is the top 25% percentile of recycling ratio, making it a clear and inrterpretable benchmark
- Converts a continuous metric into an actionable decision signal
- Enables classification modeling and operational thresholds

## 7. Modeling Approach
**Model Type**: Logistic Regression which is chosen because of:
- High interpretability
- Probabilistic outputs
- Suitable for policy and operations teams

**Key Characteristics**
- Probability-based predictions
- Threshold-controlled classification (alligned with business problem)
- Balanced class handling
- Feature scaling and encoding handled via pipeline (StandardScaler and Onehot encoding)


## 8. Evaluation metrics
Trained on 2022-01 2025-04 data, tested on 2025-05 to 2025-10 (last 6 months) 
| Metric    | Value | Business Interpretation |
|-----------|-------|--------------------------|
| Accuracy  | 93.2% | Out of all district–month observations, the model correctly predicts above/below recycling performance 93.2% of the time. |
| Recall   | 93.5% | Of all district–months that truly had **low recycling performance**, 93.5% were correctly identified by the model (most important for intervention). |
| Precision| 82.7% | When the model predicts a district is performing well, it is usually correct. |
| F1-Score | 87.8% | Balances recall and precision, showing strong overall classification performance. |
| AUC-ROC  | 0.98  | The model is very effective at separating high vs low recycling performance. |

## 9. Business Impact & Ethical Considerations
This model supports smarter decision-making for DSNY by enabling proactive action rather than reactive cleanup.

First, it helps optimize resource allocation by identifying community districts that are likely to underperform in recycling, allowing DSNY to target education campaigns, outreach, and enforcement where they are most needed.

Second, it contributes to cost savings by reducing contamination in recycling streams. By intervening early in at-risk districts, DSNY can prevent recyclable materials from being diverted to refuse, lowering processing and landfill costs.

Third, the model enables goal tracking by providing a consistent, data-driven way to monitor progress toward NYC’s Zero Waste objectives over time and across districts.

Finally, it supports equity assessment by highlighting districts that may require additional services, infrastructure, or community support—helping ensure that recycling performance improvements are citywide and equitable.

**Ethics & Impact**
- No personal or sensitive data used
- Predictions are advisory, not punitive
- Designed to guide equitable interventions

**Limitations**
- Monthly granularity limits short-term forecasting
- Population data is static (2010 Census)
- Behavioral and policy factors not included

## 10. Future Work
- Time-series forecasting for tonnage volumes
- District clustering for targeted interventions

#### *COMMENTS:*
Two target definitions were tested during model development. The recycling ratio–based classification demonstrated better interpretability and alignment with operational decision-making, and was therefore chosen for our streamlit app.

## Contributors
- [Angel Bautista](https://www.linkedin.com/in/angelgbautista/): Project manager, App Builder
- [Thierno Barry](https://www.linkedin.com/in/thierno-barry-analyst/): Python Programmer, Modeling Process

---
---
## Expanded Analysis (Individual Contribution)

This section describes the additional work I completed after the group project was finished.

### Improvements I Implemented
- Extended the dataset by adding additional dates to enable trend analysis over a longer time horizon (2022–2025).
- Created new analytical tables (`tableau_monthly_data.csv` and `tableau_district_scorecard.csv`) to support granular organic waste diversion analysis.
- Built a 5-story-point Tableau dashboard to provide a data-driven view of how organic waste management contributes to NYC's long-term sustainability targets.
- Engineered new metrics including organics capture rate, year-over-year growth percentages, performance tiers, and underperformer flags at the district level.

### Why These Changes Matter
The original group project focused on predicting recycling performance using a classification model. While effective, it did not explore the organic waste diversion story — a key pillar of NYC's Zero Waste goals. By extending the analysis with trend data, district-level scorecards, and an interactive Tableau dashboard, the expanded project now supports deeper operational insights such as identifying equity gaps across boroughs and tracking the real impact of the curbside organics rollout.

### Example New Insight
Using the expanded dataset, I found that across the same 35 fully covered districts in 2024, organics collection grew by +154% year-over-year while refuse declined by 3.5% (2024-2025 comparison) — translating to approximately 65,199 fewer tons sent to landfill. Additionally, 16 districts were identified as systemic underperformers across all recycling streams, not just organics.

---
### [Dashboard Link](https://public.tableau.com/app/profile/thierno.barry7757/viz/Book1_17627035536990/Dashboard1?publish=yes)


