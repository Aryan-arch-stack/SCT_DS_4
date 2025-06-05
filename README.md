# SCT_DS_4
1. Introduction

In recent years, road traffic accidents have emerged as a major public safety concern across the United States. With thousands of lives lost and many more injured annually, the need to understand, analyze, and predict accident severity has become increasingly important for authorities, city planners, and everyday commuters. This project aims to apply machine learning techniques and data visualization to analyze large-scale traffic accident data from across the US and develop a predictive model for accident severity. Additionally, this project includes a user-friendly web-based interface using Streamlit, enabling interactive exploration and prediction capabilities for end-users.

The dataset used in this project is the US\_Accidents\_March23.csv, which is part of a publicly available dataset sourced from various government agencies and traffic APIs. This dataset contains detailed records of road accidents reported across the United States from 2016 to early 2023. With over 2.8 million entries and more than 50 features, it offers a rich foundation for in-depth analysis. Each record provides information such as the severity of the accident, date and time, geographical location (latitude and longitude), weather conditions, visibility, road infrastructure (such as presence of a traffic signal, junction, or bump), and more. For the purposes of this project, a sample of 100,000 rows was extracted to ensure efficient processing and rapid prototyping.

The core objective of this project is threefold:

1. Severity Prediction:
   A machine learning model was trained to predict the severity of an accident (on a scale of 1 to 4) based on various features like weather condition, time of day, road visibility, temperature, and geographical indicators. By understanding these patterns, we aim to identify the key contributing factors to severe accidents and potentially aid in prevention strategies.

2. Geospatial Analysis:
   Accidents often cluster around certain high-risk zones due to poor road conditions, heavy traffic, or inadequate infrastructure. To visualize such patterns, geospatial tools like Folium were employed. Heatmaps were used to highlight accident-prone regions across the country, while KMeans clustering identified major zones where accidents are highly concentrated. This analysis can support regional transportation planning and targeted safety interventions.

3. Web Deployment using Streamlit:
   To make this solution accessible and interactive, the project includes a web-based application built using Streamlit. The app allows users to input real-world conditions (e.g., temperature, weather, road conditions) and instantly receive a severity prediction using the trained model. It also features an interactive map where users can explore accident clusters and hot zones. The web app is deployed via Ngrok, making it easy to share and demonstrate to stakeholders without requiring installation or hosting setup.


2. Dataset Overview

The dataset used for this project is the **US Accidents** dataset, which is publicly available on Kaggle. This dataset is one of the most comprehensive collections of road accident records in the United States, compiled from multiple sources such as government traffic agencies, state Departments of Transportation, and crowd-sourced platforms. It aggregates data from 2016 through March 2023, capturing the details of millions of traffic incidents nationwide.

Source and Size

The dataset comprises approximately 2.8 million records and includes over 50 columns detailing various attributes related to each accident. Due to the massive size and complexity of the full dataset, the project utilized a carefully sampled subset of 100,000 rows to enable faster analysis and iterative model development while preserving the diversity and variability present in the original data.

Data Collection and Features

The data collection methodology involves aggregating accident reports from official police and transportation databases, which are continuously updated and refined. Each record in the dataset corresponds to a unique accident event and contains a wealth of information that can be categorized broadly into the following groups:

Temporal Features:

Includes the exact date and time of the accident (`Start_Time` and `End_Time`), allowing the analysis of accident trends by hour, day of the week, month, and year. These features help identify temporal patterns, such as peak accident hours or seasonal spikes.

Geographical Features:

Each accident record contains precise latitude (`Start_Lat`) and longitude (`Start_Lng`) coordinates, enabling spatial analysis to identify accident hotspots or dangerous zones. Location-based insights are essential for regional planning and resource allocation.

Environmental and Weather Conditions:

Features such as `Weather_Condition`, `Visibility(mi)`, and `Temperature(F)` provide context about the environment during the accident. Weather conditions like rain, snow, fog, or clear skies can significantly affect road safety and accident severity. Similarly, visibility and temperature readings contribute to understanding how external factors impact driving conditions.

Road Infrastructure Attributes:

These include indicators for whether the accident occurred near a junction (`Junction`), presence of a bump (`Bump`), or if there was a traffic signal (`Traffic_Signal`). Such features help evaluate how road design and traffic control mechanisms influence accident likelihood and outcomes.

Accident Severity:

The target variable `Severity` is an ordinal variable ranging from 1 (least severe) to 4 (most severe), representing the impact of the accident, including factors such as injury level, property damage, and fatalities.

Preprocessing and Data Quality

Given the diversity and volume of the dataset, it required careful preprocessing to ensure data quality and model readiness. Missing values were addressed by removing records with null entries in essential columns such as latitude, longitude, weather condition, and start time. This ensured reliable input features and prevented noise from skewing the model.

The `Start_Time` column was converted to a datetime object to facilitate extraction of new temporal features such as the hour of the day, day of the week, and month, which are critical for uncovering time-based accident trends. Categorical variables like `Weather_Condition` were encoded to make them compatible with machine learning algorithms.

Rationale for Feature Selection

From the broad set of available features, this project focused on those most relevant to predicting accident severity and conducting geospatial analysis. Weather and visibility affect driving conditions, while temporal features help understand patterns of occurrence. Location data enables spatial clustering and mapping. Road infrastructure attributes help contextualize how environment and road design contribute to accidents.


Data Preprocessing

Data preprocessing is a crucial step in the project, ensuring that the raw dataset is cleaned, transformed, and structured appropriately for effective analysis and machine learning model development. Given the complexity and size of the US Accidents dataset, a focused approach was adopted to select and preprocess only the most relevant features that contribute to predicting accident severity and performing geospatial analysis.

Feature Selection

Key columns chosen for this project included temporal features like `Start_Time`, environmental factors such as `Weather_Condition`, and spatial coordinates (`Start_Lat`, `Start_Lng`). Additional features like `Visibility(mi)`, `Temperature(F)`, and road-related attributes (`Junction`, `Bump`, `Traffic_Signal`) were also included to enrich the dataset with contextual information influencing accident severity.

Date and Time Processing

One of the critical preprocessing tasks involved converting the `Start_Time` column from string format to a datetime object. This conversion allowed the extraction of new time-based features such as Hour of the day, Day of the week, and Month. These derived features help capture temporal patterns and trends in accident occurrences, such as identifying rush hours or seasonal effects.

Handling Missing Values

To ensure data integrity and improve model reliability, records with missing or null values in essential columns like `Start_Lat`, `Start_Lng`, `Start_Time`, and `Weather_Condition` were removed. This step minimized noise and potential errors from incomplete data.

Categorical Encoding

The `Weather_Condition` feature, being categorical with numerous unique values, was encoded using label encoding to convert it into a numerical format suitable for machine learning algorithms. This encoding preserved the categorical distinctions while enabling computational processing.



Exploratory Analysis

Exploratory Data Analysis (EDA) was conducted to understand the patterns and characteristics of the accident dataset, providing valuable insights for model building and visualization.

Geospatial Analysis: HeatMap

A heatmap visualization was created using Folium’s HeatMap plugin to highlight accident-prone regions across the United States. By plotting accident start locations, the heatmap revealed clusters of high accident density, particularly around major urban centers and highways. This spatial pattern helps identify critical areas for road safety interventions and resource allocation.

Clustering with KMeans

To further understand accident hotspots, KMeans clustering was applied to the geographic coordinates (`Start_Lat`, `Start_Lng`). The algorithm grouped the data points into distinct clusters representing accident centers. The cluster centers were then plotted on the map, marked by red circle markers. These cluster centers pinpoint key areas with a high concentration of accidents, complementing the heatmap and offering a simplified summary of accident clusters.

Weather Conditions

Weather plays a significant role in road safety. Analysis of the `Weather_Condition` feature revealed the most common weather types during accidents. Conditions such as “Clear,” “Cloudy,” “Rain,” and “Fog” were among the frequently reported states. This insight suggests that while most accidents occur in clear weather due to higher traffic volumes, adverse weather conditions still contribute significantly to accident severity and frequency.

Severity Distribution

The severity of accidents was analyzed to understand its distribution across the dataset. The majority of accidents fell under moderate severity levels, with fewer cases in extreme severity categories. This distribution helps tailor prediction models by balancing data representation across severity classes.



5. Model Building

The core objective of this project was to develop a predictive model for accident severity, which is categorized into four levels ranging from 1 (least severe) to 4 (most severe). The target variable, Severity, was treated as a multiclass classification problem.

Feature Selection

After careful preprocessing, the features selected for modeling included a mix of temporal, environmental, and road-related factors believed to influence accident severity. Key predictors used were:

  Temperature (°F): Weather temperature at the time of the accident.
  Visibility (mi): Visibility distance, which can impact driving safety.
  Junction: Whether the accident occurred near a road junction.
  Traffic\_Signal: Presence of traffic signals near the accident site.
  Bump: Road bumps presence.
  Hour, DayOfWeek, Month: Derived time features extracted from the accident start time.
  Encoded Weather\_Condition: Weather condition encoded into numerical categories.

These features were chosen to capture critical factors influencing accident dynamics.

Model Selection and Training

A Random Forest Classifier was selected due to its robustness, interpretability, and ability to handle both numerical and categorical data effectively. The dataset was split into training and testing subsets, ensuring the model was evaluated on unseen data.

Performance Evaluation

Model performance was assessed through accuracy scores and classification reports detailing precision, recall, and F1-scores for each severity class. The Random Forest model achieved a balanced performance, showing its ability to distinguish between different severity levels effectively. Confusion matrices were also analyzed to understand common misclassifications and improve model tuning iteratively.



6. Streamlit Web App

To demonstrate the practical utility of the predictive model, a Streamlit-based web application was developed to enable interactive user engagement and real-time severity prediction.

User Interface

The app provides a user-friendly interface where users can input key features relevant to an accident scenario, such as:

Time of the accident (hour, weekday)
Weather conditions (selectable categories)
Temperature and visibility readings
Road conditions like junction presence and traffic signals

These inputs are processed and fed into the pre-trained Random Forest model for prediction.

Prediction Output

Upon submitting the inputs, the app dynamically predicts the accident severity level and displays it clearly to the user, offering actionable insights about potential accident risks.

Interactive Geospatial Visualization

The web app incorporates Folium-based map visualizations showcasing:

A HeatMap highlighting accident density hotspots across the United States.
KMeans cluster centers marking significant accident clusters for easy identification of critical zones.

These visual tools enhance user understanding by providing spatial context to accident risks, making the app a comprehensive platform for accident severity prediction and spatial awareness.

Future Enhancements

The Streamlit app can be further improved by integrating live weather APIs for real-time weather conditions, adding historical trend analysis, and expanding the model to predict accident probabilities alongside severity.



6. Streamlit Web App

The culmination of this project was the development of a Streamlit web application, designed to provide an interactive and accessible platform for predicting accident severity and visualizing accident patterns geographically. Streamlit was chosen due to its simplicity, fast development cycle, and seamless integration with Python data science tools.

User Input Interface

The app allows users to input key features relevant to the prediction model. These inputs include:

Time-based inputs: Users can select the hour of the day and day of the week when the accident occurred. These temporal features are important because accident severity often varies with time — for example, night-time driving or rush hours can increase risk.

Weather conditions: A dropdown menu lets users select from typical weather conditions such as Clear, Rain, Snow, Fog, etc. Weather has a strong influence on road safety, and encoding this information into the model improves prediction accuracy.

Temperature and visibility: Users input temperature in Fahrenheit and visibility in miles, two continuous variables that affect driving conditions. Poor visibility or extreme temperatures often correlate with more severe accidents.
  
Road conditions: Binary inputs for whether the accident took place near a junction, traffic signal, or bump on the road. These factors often affect vehicle behavior and accident outcomes.

By allowing users to interactively specify these conditions, the app makes the machine learning model accessible for real-world what-if analyses.

Severity Prediction

Once the user inputs are provided, the app preprocesses the data in the same way as the training pipeline—encoding categorical variables and scaling where necessary—before feeding it to the trained Random Forest Classifier. The model outputs the predicted severity class (1 through 4), along with a brief description of what the severity level indicates.

This immediate feedback provides an intuitive understanding of how different conditions might influence accident severity, useful for drivers, planners, and researchers alike.

Interactive Geospatial Visualizations

The app includes advanced visualization features powered by Folium, a Python library for interactive leaflet maps:

Heatmap Visualization: A heatmap layer shows concentrations of accidents across the United States, highlighting high-density accident zones. This visual helps users identify geographic hotspots where accidents frequently occur.
  
Cluster Centers: Using KMeans clustering, accident locations are grouped into clusters, and cluster centers are marked on the map with circle markers. This helps identify critical regions for targeted road safety interventions.

Together, these maps provide spatial context that complements the severity predictions, supporting informed decision-making for road safety authorities and policymakers.

Benefits of the Web App

  Accessibility: Anyone with internet access can use the app, without needing deep technical expertise.
  Real-time Analysis: Users can input custom scenarios and immediately see severity predictions.
  Visualization: Maps enhance spatial awareness, important for understanding accident risk distribution.
  Scalability: The app framework supports future extensions such as real-time weather integration and live traffic data.



7. Challenges

During the course of this project, several challenges were encountered that are worth discussing, as they highlight common obstacles in real-world data science and machine learning workflows, especially in the domain of accident analysis.

Large Dataset Size

The full dataset contains approximately 2.8 million records with over 50 columns, spanning many years of accident data. Handling such a large volume of data required:

  Sampling: Due to computational and memory constraints, only a subset of 100,000 rows was sampled for model training and initial exploratory analysis.
  Performance trade-offs: Sampling helps reduce runtime but risks missing patterns in less frequent classes or rare conditions. Balancing dataset size and model quality was an ongoing consideration.
  Tooling: Using libraries like Dask helped read large CSV files efficiently but model training still needed smaller data chunks.

Unbalanced Severity Labels

Accident severity classes are naturally imbalanced, with certain severity levels (e.g., minor accidents) occurring more frequently than severe or fatal ones. This imbalance poses difficulties such as:

Biased models: Without addressing class imbalance, the model may be biased towards predicting the majority classes.
Evaluation difficulties: Accuracy alone can be misleading; metrics like precision, recall, and F1-score per class were necessary to assess true performance.
Potential remedies: Techniques like resampling, class weighting, or synthetic data generation (SMOTE) could be explored for better model fairness.

Missing and Noisy Data

Real-world datasets often suffer from:

 Missing values: Particularly in key columns such as weather conditions or location coordinates.
 Noisy entries: Inconsistent formats, outliers, or incorrect entries that can mislead the model.

To mitigate these issues:

  Rows with critical missing data were dropped.
  Categorical variables were cleaned and encoded properly.
  Data types were standardized, such as converting timestamps to datetime objects and extracting time features.

Still, data quality challenges inevitably limited the model's predictive accuracy and required careful preprocessing.



8. Conclusion

This project demonstrates the power and potential of machine learning (ML) combined with geospatial analysis to better understand and predict the severity of road accidents. By leveraging a large, real-world dataset and focusing on critical factors such as weather, time, and location, the Random Forest model developed here can predict accident severity with reasonable accuracy.

Key Takeaways

Predictive Modeling: ML models can successfully learn complex relationships between environmental, temporal, and road-related factors to predict accident severity. Such models can be invaluable tools for risk assessment and preventive measures.
  
Importance of Visualization: Geospatial heatmaps and clustering provide essential context by highlighting high-risk zones. These insights enable targeted interventions like improved signage, road repairs, or increased patrols in accident-prone areas.

Interactive Tools for Impact: The Streamlit web app bridges the gap between data science and end users by providing an easy-to-use interface for prediction and visualization. This accessibility is vital for spreading awareness and informing decision-making beyond data scientists.

Future Directions

The project lays the groundwork for several exciting enhancements:

  Real-Time Data Integration: Linking live weather, traffic, and incident reports could enable dynamic risk predictions.
  Expanded Features: Incorporating driver behavior data, vehicle types, and road infrastructure details could improve model granularity.
  Advanced Models: Experimenting with deep learning or ensemble techniques may boost predictive accuracy.
  Policy Applications: Collaborating with transportation authorities to deploy the app for real-world monitoring and preventive planning.

In conclusion, by combining machine learning, data preprocessing, geospatial visualization, and user-friendly web deployment, this project offers a practical framework for accident severity prediction. It highlights the critical role of data-driven insights in enhancing road safety and ultimately saving lives.
