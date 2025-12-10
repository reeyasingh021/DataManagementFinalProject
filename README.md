# NYC Real Estate Price Prediction with Crime Data

A machine learning project that predicts NYC property prices by analyzing the impact of local crime statistics using geospatial data analysis and XGBoost modeling.

**Team Members:** Chloe Callueng, Isha Jain, Reeya Singh

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Data Sources](#data-sources)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Overview

This project addresses a critical gap in real estate analytics: **quantifying how neighborhood crime affects individual property values in NYC**. While buyers can access crime statistics and property prices separately, no robust tool exists to measure their relationship quantitatively.

We built a predictive model that:
- Integrates geospatial crime data with property listings
- Uses spatial joins to calculate crime statistics within a 0.5-mile radius of each property
- Predicts property prices with 80% accuracy (R² = 0.80)
- Reveals that crime severity (felony percentage) impacts prices more than total crime volume

**🎨 [View Interactive Demo](https://claude.ai/public/artifacts/16eb3a0b-d3af-406c-bbe9-75d1656f5b54)** - Try out the prediction interface!

---

## 🎯 Project Goals

1. **Data Integration:** Combine NYC housing market data with NYPD arrest records using spatial joins
2. **Feature Engineering:** Calculate crime statistics (density, felony rates, recent trends) for each property's neighborhood
3. **Predictive Modeling:** Build an XGBoost model that predicts property prices incorporating crime data
4. **Insight Generation:** Identify which crime factors most significantly impact property valuations

---

## 📊 Data Sources

### NYC Housing Market Dataset
- **Source:** [Kaggle - New York Housing Market](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)
- **Records:** 4,801 properties (4,585 after cleaning)
- **Features:** Property type, price, beds, baths, square footage, location (lat/lon), borough, ZIP code

### NYPD Arrests Dataset
- **Source:** [Kaggle - NYPD Arrests Data](https://www.kaggle.com/datasets/danilzyryanov/arrests-data-by-new-york-police-department)
- **Records:** 309,355 arrest records (2006-2017)
- **Features:** Arrest date, offense category, law classification (Felony/Misdemeanor/Violation), location, demographics

---

## ✨ Features

### Data Processing
- **Comprehensive data cleaning:** Removed duplicates, handled outliers, standardized formats
- **Geospatial analysis:** Haversine distance calculations for accurate crime proximity
- **Feature engineering:** Created 11 crime-related features per property

### Crime Features Calculated (0.5-mile radius per property)
- Total crime count
- Crime density (crimes per square mile)
- Felony/misdemeanor/violation counts and percentages
- Recent crime trends (last 90 days)
- Average and minimum distance to crimes

### Database Design
- **Normalized schema** with 4 tables:
  - `properties` - Property details with unique IDs
  - `crimes` - Individual crime records
  - `property_crime_stats` - Aggregated crime features (linked via foreign key)
  - `borough_stats` - Borough-level summaries

### Machine Learning Model
- **Algorithm:** XGBoost Regressor
- **Performance:** R² = 0.80, RMSE = 0.48 (log scale)
- **Optimization:** RandomizedSearchCV for hyperparameter tuning
- **Interpretability:** SHAP values for feature importance analysis

---

## 🛠 Technical Stack

- **Languages:** Python 3.12
- **Libraries:**
  - Data Processing: `pandas`, `numpy`
  - Geospatial Analysis: Custom haversine implementation
  - Machine Learning: `xgboost`, `scikit-learn`
  - Database: `sqlite3`
  - Visualization: `matplotlib`, `shap`
  - API Development: `flask`, `flask-cors`, `pyngrok`
  - UI: `gradio` (for Colab interface)

- **Environment:** Google Colab

---

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost sqlite3 matplotlib shap flask flask-cors pyngrok gradio
```

### Download Datasets
1. **NYC Housing Dataset** - [Download from Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)
   - Included in repository: `NY-House-Dataset.csv`
2. **NYPD Arrests Dataset** - [Download from Kaggle](https://www.kaggle.com/datasets/danilzyryanov/arrests-data-by-new-york-police-department)
   - ⚠️ **Too large for GitHub** - Download separately and place in project directory
   - Cleaned version available: `clean_data2.csv`

---

## 📁 Project Structure

```
DataManagementFinalProject/
│
├── DataManagement_Project.ipynb    # Main Jupyter notebook
├── Data Management Project.py      # Python script version of notebook
├── README.md                        # This file
├── NY-House-Dataset.csv            # Housing data (4,801 properties)
├── clean_data2.csv                 # Cleaned crime data (309,355 arrests)
│
└── Note: NYPD_Arrests.csv is too large for GitHub (>100MB)
    Download from: https://www.kaggle.com/datasets/danilzyryanov/arrests-data-by-new-york-police-department
```

---

## 🚀 Usage

### Quick Start with Google Colab

1. **Upload datasets to Colab:**
   - `NY-House-Dataset.csv` (included in repo)
   - `NYPD_Arrests.csv` (download from Kaggle)

2. **Open and run `DataManagement_Project.ipynb`**

3. **Follow the sections in order:**
   - Data Cleaning
   - Spatial Joins
   - Database Creation
   - Model Training
   - Visualization

### Interactive Demo

Try the prediction interface: **[NYC Property Price Predictor Demo](https://claude.ai/public/artifacts/16eb3a0b-d3af-406c-bbe9-75d1656f5b54)**

### Detailed Steps

#### 1. Data Cleaning & Preparation

Run the data cleaning sections in the notebook:
- Removes duplicates and outliers
- Standardizes borough names
- Extracts ZIP codes
- Creates unique property IDs

### 2. Spatial Join

Execute the spatial join to calculate crime features:
```python
crime_features = create_spatial_join(
    clean1,           # Housing dataframe
    data2,            # Crime dataframe
    radius_miles=0.5, # Search radius
    sample_size=None  # Process all properties
)
```

**Processing time:** ~2-3 hours for full dataset (4,585 properties)

### 3. Database Creation

Build the normalized database:
```python
db_path = create_database_schema(
    clean1, 
    data2, 
    crime_features,
    db_path='nyc_real_estate_crime.db'
)
```

### 4. Train Prediction Model

Train the XGBoost model:
```python
# Load data from database
conn = sqlite3.connect('nyc_real_estate_crime.db')
train = pd.read_sql_query("SELECT * FROM properties_with_crime;", conn)

# Train model (see notebook for full code)
xgbmod = XGBRegressor(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.8
)
xgbmod.fit(X_train, Y_train)
```

### 5. Make Predictions

Use the trained model to predict property prices:
```python
# Example: Predict price for a 2-bed, 2-bath condo in Manhattan
prediction = xgbmod.predict(input_features)
price = np.exp(prediction)  # Convert from log scale
```

Or try the **[Interactive Demo](https://claude.ai/public/artifacts/16eb3a0b-d3af-406c-bbe9-75d1656f5b54)** to see predictions in action!

---

## 📈 Results

### Model Performance
- **R² Score:** 0.80 (exceeds target of 0.70)
- **RMSE:** 0.48 (on log-transformed price)
- **MAE:** 0.30 (on log-transformed price)

### Visualizations

The project includes several visualizations analyzing the relationship between crime and property values:
- Price vs Total Crimes scatter plot
- Price vs Crime Density analysis  
- Average Property Price by Borough
- Average Crimes per Property by Borough
- Beds vs Price correlation
- Recent Crimes (90 days) vs Price impact

### Key Findings

1. **Crime severity matters more than volume**
   - Felony percentage has greater impact than total crime count
   - Recent crimes (last 90 days) significantly influence prices

2. **Property characteristics are primary drivers**
   - Number of bathrooms is the strongest predictor
   - Property size and bedrooms follow closely
   - Borough location plays a significant role

3. **Crime impact varies by neighborhood**
   - Manhattan properties: Average 1,853 nearby crimes (0.5mi radius)
   - Staten Island properties: Average 91 nearby crimes
   - High crime density can reduce property values by 5-15%

### SHAP Analysis Results

Feature importance ranking:
1. BATH (bathrooms)
2. PROPERTYSQFT (property size)
3. BEDS (bedrooms)
4. BOROUGH (Manhattan has highest impact)
5. felony_pct (crime severity)
6. crimes_last_90_days (recent trend)
7. violation_pct
8. misdemeanor_pct

---

## 🔮 Future Enhancements

### Short-term
- [ ] Deploy interactive web application for real-time predictions
- [ ] Add visualization dashboard with crime heatmaps
- [ ] Implement price change prediction over time
- [ ] Cross-validate with additional years of data

### Long-term
- [ ] Incorporate additional features:
  - School ratings
  - Public transportation access
  - Amenities (parks, restaurants)
  - Income demographics
- [ ] Expand to other major cities
- [ ] Build recommendation system for undervalued properties
- [ ] Analyze temporal trends (pre/post-pandemic effects)

### Technical Improvements
- [ ] Optimize spatial join processing (implement R-tree indexing)
- [ ] Add automated data pipeline for regular updates
- [ ] Create Docker container for reproducibility
- [ ] Implement model versioning and monitoring

---

## 📊 Database Schema

### Tables

**1. properties**
```sql
PROPERTY_ID (PK), TYPE, PRICE, BEDS, BATH, PROPERTYSQFT, 
BOROUGH, LATITUDE, LONGITUDE, ZIP_CODE
```

**2. crimes**
```sql
CRIME_ID (PK), ARREST_DATE, LAW_CODE, LAW_CATEGORY, 
BOROUGH, Latitude, Longitude, AGE_GROUP, PERP_RACE
```

**3. property_crime_stats**
```sql
PROPERTY_ID (PK, FK), total_crimes, crime_density, 
felony_count, misdemeanor_count, violation_count, 
felony_pct, misdemeanor_pct, violation_pct, 
crimes_last_90_days, avg_crime_distance, min_crime_distance
```

**4. borough_stats**
```sql
BOROUGH (PK), total_properties, avg_price, median_price, 
total_crimes, avg_crimes_per_property, avg_crime_density, 
avg_felony_pct
```

---

## 🤝 Contributing

This is an academic project. For questions or collaboration opportunities, please contact the team members.

---

## 📄 License

This project is part of academic coursework. All data sources are publicly available and properly attributed.

---

## 🙏 Acknowledgments

- **Data Sources:** Kaggle contributors for NYC Housing and NYPD datasets
- **Libraries:** Scikit-learn, XGBoost, SHAP, and the Python data science community
- **Course:** Data Management for Data Science

---

## 📞 Contact

**Team Members:**
- Chloe Callueng
- Isha Jain
- Reeya Singh

**Project Repository:** https://github.com/reeyasingh021/DataManagementFinalProject

**Interactive Demo:** https://claude.ai/public/artifacts/16eb3a0b-d3af-406c-bbe9-75d1656f5b54

---

*Last Updated: December 2025*
