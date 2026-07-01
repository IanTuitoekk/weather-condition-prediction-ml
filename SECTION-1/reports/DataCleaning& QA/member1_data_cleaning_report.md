# Member 1: Data Cleaning & Quality Assessment Report

## Dataset Description

The Global Weather Repository dataset was used in this project to analyze weather conditions and develop predictive machine learning models. The dataset contains weather observations collected from multiple locations worldwide and includes meteorological, environmental, and astronomical variables.

The dataset consists of **120,627 records** and **41 features**. The attributes include both numerical and categorical variables.

Examples of variables contained in the dataset include:

* Country and location information
* Temperature (Celsius and Fahrenheit)
* Weather condition descriptions
* Wind speed and direction
* Atmospheric pressure
* Humidity and cloud cover
* Precipitation measurements
* Visibility and UV index
* Air quality indicators
* Sunrise, sunset, moonrise, and moonset information

The dataset was inspected to assess data quality before performing exploratory analysis and machine learning tasks.

## Data Cleaning

### Missing Value Analysis

The dataset was examined for missing values using the Pandas `isnull()` function.

**Result:**

* Total missing values detected: **0**
* All attributes contained complete observations.

A missing-value visualization was generated to confirm the absence of missing data.

### Duplicate Record Analysis

Duplicate records were identified using the `duplicated()` function.

**Result:**

* Duplicate rows detected: **0**

Since no duplicate records were found, no rows were removed from the dataset.

### Data Type Verification

The data types of all features were verified to ensure consistency.

* Numerical variables such as temperature, humidity, pressure, and wind speed were stored using numeric data types (`int64` and `float64`).
* Categorical variables such as country, timezone, weather condition, and moon phase were stored as string/object data types.

The dataset showed consistent and appropriate data types across all variables.

### Outlier Detection

Outliers were identified using the Interquartile Range (IQR) method for numerical features.

Boxplots were generated to visualize the distribution of selected variables and identify extreme values.

Several variables, including temperature, wind speed, atmospheric pressure, precipitation, and air quality measurements, contained outliers.

These observations were retained because extreme weather conditions are realistic occurrences and may provide valuable information for predictive modeling.

### Final Dataset

After performing data quality assessment and cleaning procedures, the dataset was determined to be suitable for further analysis.

Key findings include:

* No missing values were present.
* No duplicate records were detected.
* Data types were consistent.
* Outliers were identified and retained due to their practical significance in weather analysis.

The cleaned dataset was saved as:

`data/GlobalWeatherRepository_cleaned.csv`

This dataset will be used in subsequent stages, including exploratory data analysis, feature engineering, and machine learning model development.
