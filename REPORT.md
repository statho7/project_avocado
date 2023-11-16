# 1. Introduction
## 1.1 Introduction to Avocado dataset.
The Hass Avocado Board website provided the data for this analysis, which was downloaded in May 2018. The dataset contains weekly retail scanned data for National retail volume (units) and price between 2015 and 2018. The data includes information on the average price of a single avocado, the total number of avocados sold, and the total number of bags of avocados sold, among other features. The dataset also includes information on the type of avocado (conventional or organic), the year of the observation, and the city or region of the observation.
## 1.2 Explain the objective or problem statement of your analysis.
The primary objective of this analysis is to thoroughly examine and understand the Avocado dataset. This involves preprocessing the data to ensure it is suitable for analysis, which may include handling missing values, normalizing and standardizing the data, and potentially engineering new features from existing ones. Once the data is prepared, the goal is to create time series forecasting models. These models will allow us to predict future trends based on the historical data provided in the dataset. The ultimate aim is to gain insights that could be valuable for decision-making processes related to avocado sales.

# 2. Data Understanding
## 2.1 Describe the dataset's features, including their types and significance.
GitHub Copilot: Here's a suggestion for describing the features of the dataset:

The dataset comprises several features, each with its unique significance:

- **Unnamed column**: This integer feature represents the number of weeks left for the end of the year of the observation. It provides a temporal context to each observation.
- **Date**: This datetime feature is the date of the observation, providing a precise point in time for each data entry.
- **AveragePrice**: This continuous feature represents the average price of a single avocado, a key indicator of market conditions.
- **Total Volume**: This continuous feature represents the total number of avocados sold, providing a measure of overall demand.
- **4046, 4225, 4770**: These continuous features represent the total number of avocados with PLU 4046, 4225, and 4770 sold, respectively. They provide insight into the demand for specific avocado types.
- **Total Bags, Small Bags, Large Bags, XLarge Bags**: These continuous features represent the total number of bags (and their sizes) of avocados sold, providing insight into packaging preferences.
- **type**: This categorical feature indicates whether the avocado is conventional or organic, providing insight into consumer preferences.
- **year**: This integer feature represents the year of the observation, allowing for analysis across different time periods.
- **region**: This categorical feature represents the city or region of the observation, allowing for regional comparisons and insights.

<!-- ## 2.2 Discuss any challenges or anomalies encountered during data collection or understanding. -->

# 3. Exploratory Data Analysis (EDA)
## 3.1 Summary statistics
Based on our investigation for the TotalUS region we will provide a summary of the statistics for it for the whole dataset and then specifically for organic and then conventional avocado sales.

### 3.1.1 Combined statistics
The average price of avocados stands at about $1.32, showcasing a standard deviation of 0.30, with prices ranging from $0.76 to $2.09. The dataset records an average sale of approximately 17.35 million avocados, fluctuating between 501,815 and 62.51 million units, with a standard deviation of around 16.97 million units. Variations in demand among avocado types, represented by PLU codes 4046, 4225, and 4770, exhibit diverse mean and standard deviation values. Similarly, consumer preferences for bag sizes—total bags, small bags, large bags, and XLarge bags—are highlighted by varying mean and standard deviation figures for each bag category.

### 3.1.2 Organic avocados
The average price of avocados in this dataset is approximately $1.55, with prices ranging between $1.00 and $2.09 and showing a standard deviation of 0.20. On average, about 967,566 avocados are sold, with sales volumes varying from approximately 501,815 to 1,814,930 units, displaying a standard deviation of roughly 302,482 units. Notably, the sales of avocados categorized under PLU codes 4046, 4225, and 4770 demonstrate distinct mean values and variations, signaling varying demand levels for each type. Moreover, consumer preferences for different bag sizes (total bags, small bags, large bags, and XLarge bags) are evident in the fluctuating mean and standard deviation values, highlighting varying choices in avocado packaging sizes.

### 3.1.3 Conventional avocados
The average price of avocados stands at around $1.09, with slight fluctuations between $0.76 and $1.65. In terms of sales volume, approximately 33,735,040 avocados are typically sold, ranging from 21,009,730 to 62,505,650 units. Variations in demand are evident across different avocado types indicated by PLU codes 4046, 4225, and 4770, each showing distinct average sales and fluctuations. Similarly, consumer preferences for avocado packaging sizes differ, as seen in the varying averages and fluctuations of total bags, small bags, large bags, and XLarge bags sold.

## 3.2 Data visualization: Histograms, box plots, correlation matrices, etc.
## 3.3 Insights gained from EDA, trends, patterns, and initial observations.

# 4. Data Preprocessing
## 4.1 Data normalization and standardization techniques used.
## 4.2 Feature engineering, if performed (creating new features from existing ones).
The first feature that was added was the weeks left until the end of the year. The unnamed column was used as it was the count of weeks that passed since the start of the year. This was decided to be able and plot on the same plot what happened to features like Average Price and Total Volume of avocados sold by comparing the years 2015, 2016 and 2017.

The second feature that was added was the Total Sales which was computed by multiplying the Average Price and Total Volume of each observation.

# 5. Analysis and Modeling
## 5.1 Correlation analysis: Explore relationships between variables.

### 5.1.1 Correlation Analysis for TotalUS region and conventional avocados
While examining the TotalUS region for conventional avocados strong negative correlations were observed between the AveragePrice and the sales of avocado types 4225 and 4770, with values of -0.72 and -0.65, respectively. Additionally, a moderate negative correlation of -0.51 is noted between AveragePrice and the Total Volume sold, indicating a tendency for volume and average price to move inversely. A moderate negative relationship of -0.44 is also identified between the AveragePrice and sales of type 4046 avocados(small avocados). Conversely, weaker negative correlations, nearly negligible at -0.10, exist between sales of type 4770 avocados and the usage of Large Bags for packaging.

On the other hand, the dataset reveals strong positive correlations between certain variables. Specifically, an important positive relationship of 0.73 is evident between the Total Volume of avocados sold and the sales of type 4225(large avocado), indicating a significant association between these two factors. Additionally, a robust positive correlation of 0.83 is observed between Total Volume and sales of type 4046 avocados. The packaging categories, Small Bags and Large Bags, demonstrate a strong positive relationship at 0.84, suggesting a notable association between sales in these different bag sizes. Moreover, a substantial positive correlation of 0.92 exists between the usage of Large Bags and the overall Total Bags sold, as well as between Total Bags and Large Bags, emphasizing the close relationship between these variables. Finally, the strongest positive correlation of 0.99 is noted between Total Bags and Small Bags, indicating an almost perfect relationship between total bag sales and sales specifically in small bags.

### 5.1.2 Correlation Analysis for TotalUS region and organic avocados
Examining the correlations within the dataset for organic avocados in TotalUS region uncovers several noteworthy relationships among the avocado-related variables. Notably, negative correlations are identified between 4770 avocados and various bag-related categories. A negative correlation of -0.44 exists between 4770 avocados and Small Bags, suggesting a moderate inverse relationship. Similarly, this same strength of negative correlation, -0.39, is observed between 4770 avocados and AveragePrice, implying a moderate inverse movement between the sales of type 4770 avocados and their pricing. Additionally, negative correlations of -0.37 and -0.36 are noted respectively between 4770 avocados and Total Bags and XLarge Bags, indicating a moderate inverse relationship between sales of type 4770 avocados and the number of total bags and XLarge bags used for packaging. A negative correlation of -0.34 is also observed between 4046 avocados and Total Bags, indicating a moderate inverse relationship between sales of type 4046 avocados and the total bags used for packaging.

Conversely, strong positive correlations are apparent between various bag-related categories and avocado sales. A notable positive correlation of 0.66 exists between Total Bags and Large Bags, emphasizing a significant relationship between the total bags sold and the specific sales in large bags. Similarly, a positive correlation of 0.67 is observed between Large Bags and Total Volume as well as between Total Volume and Large Bags, indicating a considerable relationship between the number of large bags used and the total volume of avocados sold. The strongest positive correlations are observed between Total Bags and Small Bags, with a value of 0.95, emphasizing an almost perfect relationship between the total bags sold and sales specifically in small bags. Moreover, a robust positive correlation of 0.93 is noted between Total Bags and Total Volume, suggesting a significant association between the total bags sold and the total volume of avocados sold.

## 5.2 Outlier detection and treatment methods used.
## 5.3 Description and implementation of the Prophet model and XGBoost models.

# 6. Model Evaluation
## 6.1 Evaluation metrics used for the Prophet and XGBoost models (e.g., RMSE, MAE, accuracy, etc.).
## 6.2 Comparison of model performance, strengths, and weaknesses.

# 7. Results and Interpretation
## 7.1 Key findings from the analysis.
## 7.2 Insights derived from the models and their implications.

# 8. Conclusion
## 8.1 Summarize the project's goals and achievements.
## 8.2 Discuss the significance of your findings and their real-world implications.
