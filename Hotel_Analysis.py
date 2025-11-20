#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-20T12:17:54.947Z
"""

# Import Libraries
import pandas as pd             # For data loading, cleaning, manipulation
import numpy as np              # For numerical operations
import matplotlib.pyplot as plt # For basic visualizations
import seaborn as sns           # For statistical plots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# **Phase 01 - Data Cleaning and EDA**


# Load Dataset 
hotels_df = pd.read_csv( r"D:\ITM_Term5\Hackathon\Internal_Hack\hotels.csv", encoding="latin1")
display(hotels_df.head())

# Display the Number of Records and Columns
print('Rows, Columns:', hotels_df.shape)

# Display Column_Name, Datatype, Null Values
hotels_df.info()

# Total NULL values in each column
hotels_df.isnull().sum()

# Strip whitespaces & standardize column names
hotels_df.columns = hotels_df.columns.str.strip()

# **Remove Unwanted Columns from Dataset**


# Remove unwanted columns
columns_to_drop = ['FaxNumber', 'PhoneNumber', 'PinCode']
hotels_df = hotels_df.drop(columns=columns_to_drop, axis=1)

# Verify
hotels_df.info()

# **1. Missing Values in cityCode Column**


# Show the countyNames that still have missing countyCode
hotels_df.loc[hotels_df['countyCode'].isnull(), 'countyName'].unique()

# Fill countyName
hotels_df.loc[hotels_df['countyName'] == 'Namibia', 'countyCode'] = 'NAM'

hotels_df["countyCode"].isnull().sum()

# **2. Missing Values in Address Column**


# Step 1: Find rows where Address is missing
missing_address = hotels_df[hotels_df['Address'].isnull()]

# Step 2: Get unique HotelCodes with missing Address
missing_address_hotelcodes = missing_address['HotelCode'].unique()
print(missing_address_hotelcodes)

# List of HotelCodes with missing Address (from previous output)
hotel_codes_missing_address = [
    1398355, 1051847, 1566286, 1536721, 1112967, 1131921, 1108496, 1110242,
    1110735, 1108909, 1108935, 1111614, 1106476, 1106544, 1106660, 1106864,
    1130345, 1106478, 1106659, 1006670, 1109810, 1110492, 1010270, 1217637,
    1013142, 1134396, 1113905, 1107943, 1213576, 1003399, 1111974, 1113598,
    1201653, 1111464, 1111609, 1111980, 1112037, 1112300, 1112386, 1112388,
    1020088, 1207945, 1307552, 1019054, 1210583, 1107571, 1112435, 1008694,
    1132809, 1008206, 1109556, 1109561, 1109873, 1110092, 1111254, 1016645,
    1106467, 1106607, 1106820, 1106929, 1107211, 1113773, 1005076, 1109057,
    1109065, 1109216, 1110085, 1110120, 1111037, 1111148, 1111938, 1113833,
    1114711, 1000704, 1212316, 1160492, 1211837, 1019303, 1003236, 1109403,
    1188271, 1114603, 1212869, 1108421, 1109087, 1215964, 5018207, 1074241,
    1513906, 1942791, 1146786, 1916793, 1196861, 1947599, 1063131, 1086480, 1079888
]

# Check if these HotelCodes have any non-missing Address in hotels_df
missing_address_check = hotels_df[hotels_df['HotelCode'].isin(hotel_codes_missing_address)].groupby('HotelCode')['Address'].apply(lambda x: x.notnull().any())

print(missing_address_check)

# Fill these missing addresses using placeholder 'Not Available'
hotels_df['Address'] = hotels_df['Address'].fillna('Not Available')

# Verify if all missing 'Address' values have been filled
print(f"Number of missing 'Address' values after filling: {hotels_df['Address'].isnull().sum()}")

# **3. Missing Values in Attractions Column**


# Step 1: Find rows where Attractions is missing
missing_attractions = hotels_df[hotels_df['Attractions'].isnull()]

# Step 2: Get unique HotelCodes with missing Attractions
missing_attractions_hotelcodes = missing_attractions['HotelCode'].unique()
print(missing_attractions_hotelcodes)

# Check if any row for each HotelCode has a non-missing Attractions
attractions_check = hotels_df[hotels_df['HotelCode'].isin(missing_attractions_hotelcodes)].groupby('HotelCode')['Attractions'].apply(lambda x: x.notnull().any())
print(attractions_check)

# Fill these missing attractions using placeholder 'Not Available'
hotels_df['Attractions'] = hotels_df['Attractions'].fillna('Not Available')

# Verify if all missing 'Attractions' values have been filled
print(f"Number of missing 'Attractions' values after filling: {hotels_df['Attractions'].isnull().sum()}")

# **4. Missing Values in Description Column**


# Rows where Description is missing
missing_description = hotels_df[hotels_df['Description'].isnull()]

# Unique HotelCodes with missing Description
missing_description_hotelcodes = missing_description['HotelCode'].unique()
print(missing_description_hotelcodes)

# Check if any row per HotelCode has a non-missing Description
description_check = hotels_df[hotels_df['HotelCode'].isin(missing_description_hotelcodes)].groupby('HotelCode')['Description'].apply(lambda x: x.notnull().any())
print(description_check)

# Fill these missing Description using placeholder 'No Description'
hotels_df['Description'] = hotels_df['Description'].fillna('No Description')

# Verify if all missing 'Description' values have been filled
print(f"Number of missing 'Description' values after filling: {hotels_df['Description'].isnull().sum()}")

# **5. Missing Values in HotelFacilities Column**


# Rows where HotelFacilities is missing
missing_facilities = hotels_df[hotels_df['HotelFacilities'].isnull()]

# Unique HotelCodes with missing HotelFacilities
missing_facilities_hotelcodes = missing_facilities['HotelCode'].unique()
print(missing_facilities_hotelcodes)

# Check if any row per HotelCode has a non-missing HotelFacilities
facilities_check = hotels_df[hotels_df['HotelCode'].isin(missing_facilities_hotelcodes)].groupby('HotelCode')['HotelFacilities'].apply(lambda x: x.notnull().any())
print(facilities_check)

# Fill these missing HotelFacilities using placeholder 'No Facilities Info'
hotels_df['HotelFacilities'] = hotels_df['HotelFacilities'].fillna('No Facilities Info')

# Verify if all missing 'HotelFacilities' values have been filled
print(f"Number of missing 'HotelFacilities' values after filling: {hotels_df['HotelFacilities'].isnull().sum()}")

# **6. Missing Values in HotelWebsiteURL**


# Rows where HotelWebsiteUrl is missing
missing_website = hotels_df[hotels_df['HotelWebsiteUrl'].isnull()]

# Unique HotelCodes with missing HotelWebsiteUrl
missing_website_hotelcodes = missing_website['HotelCode'].unique()
print(missing_website_hotelcodes)

# Check if any row per HotelCode has a non-missing HotelWebsiteUrl
website_check = hotels_df[hotels_df['HotelCode'].isin(missing_website_hotelcodes)].groupby('HotelCode')['HotelWebsiteUrl'].apply(lambda x: x.notnull().any())
print(website_check)

# Fill these missing HotelWebsiteUrl using placeholder 'No Website'
hotels_df['HotelWebsiteUrl'] = hotels_df['HotelWebsiteUrl'].fillna('No Website')

# Verify if all missing 'HotelWebsiteUrl' values have been filled
print(f"Number of missing 'HotelWebsiteUrl' values after filling: {hotels_df['HotelWebsiteUrl'].isnull().sum()}")

# **7. Missing Values in Map Column**


# Rows where Map is missing
missing_map = hotels_df[hotels_df['Map'].isnull()]

# Unique HotelCodes with missing Map
missing_map_hotelcodes = missing_map['HotelCode'].unique()
print(missing_map_hotelcodes)

# Check if any row per HotelCode has a non-missing Map
map_check = hotels_df[hotels_df['HotelCode'].isin(missing_map_hotelcodes)].groupby('HotelCode')['Map'].apply(lambda x: x.notnull().any())
print(map_check)

# Fill these missing Map using placeholder 'No Map'
hotels_df['Map'] = hotels_df['Map'].fillna('No Map')

# Verify if all missing 'Map' values have been filled
print(f"Number of missing 'Map' values after filling: {hotels_df['Map'].isnull().sum()}")

# **8. Final Check Missing Values = 0;**


hotels_df.isnull().sum()

# **Exploratory Analysis**


# TOP 10 COUNTRIES
top_countries = (hotels_df['countyName']
                 .value_counts()
                 .head(10)
                 .reset_index())
top_countries.columns = ['Country Name', 'Number Of Hotels']

plt.figure(figsize=(12, 6))
plt.barh(top_countries['Country Name'], top_countries['Number Of Hotels'])
plt.xlabel("Number of Hotels")
plt.ylabel("Country Name")
plt.title("Top 10 Countries with the Most Hotels")
plt.gca().invert_yaxis()  # Highest value on top
plt.tight_layout()
plt.show()


# TOP 10 CITIES 
top_cities = (hotels_df['cityName']
              .value_counts()
              .head(10)
              .reset_index())
top_cities.columns = ['City Name', 'Number Of Hotels']

plt.figure(figsize=(12, 6))
plt.barh(top_cities['City Name'], top_cities['Number Of Hotels'])
plt.xlabel("Number of Hotels")
plt.ylabel("City Name")
plt.title("Top 10 Cities with the Most Hotels")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# **Interpretation**
# 
# 1.	Top 10 Counties based on Hotel Count
# -	These countries represent large tourism economies.
# -	They have high hotel supply, which can mean:
# •	high competition
# •	presence of Premium, luxury, & budget hotels
# •	strong tourism infrastructure
# 
# 2.	Top 10 Cities based on Hotel Count
# -	The cities with highest hotel counts are top global tourist hotspots.
# -	Many of them (London, Orlando, Bali, Rome, Phuket) have:
# •	High domestic & international tourism
# •	Thousands of hotels competing
# 
# -	These are oversupplied markets → very competitive.
# 


# TOP 10 COUNTRIES WITH LEAST HOTELS ----------
least_countries = (hotels_df['countyName']
                   .value_counts()
                   .tail(10)            # Bottom 10
                   .reset_index())
least_countries.columns = ['Country Name', 'Number Of Hotels']

plt.figure(figsize=(12, 6))
plt.barh(least_countries['Country Name'], least_countries['Number Of Hotels'])
plt.xlabel("Number of Hotels")
plt.ylabel("Country Name")
plt.title("Top 10 Countries with the Least Number of Hotels (Undersupply)")
plt.tight_layout()
plt.show()


# TOP 10 CITIES WITH LEAST HOTELS ----------
least_cities = (hotels_df['cityName']
                .value_counts()
                .tail(10)              # Bottom 10
                .reset_index())
least_cities.columns = ['City Name', 'Number Of Hotels']

plt.figure(figsize=(12, 6))
plt.barh(least_cities['City Name'], least_cities['Number Of Hotels'])
plt.xlabel("Number of Hotels")
plt.ylabel("City Name")
plt.title("Top 10 Cities with the Least Number of Hotels (Undersupply)")
plt.tight_layout()
plt.show()

# **Interpretation** 
# 
# - Cities/Countries with few hotels = Potential Opportunities but only if attractions and travel demand exist
# 
# 
# | Category                           | Meaning                       |
# | ---------------------------------- | ----------------------------- |
# | **High Attractions + Low Hotels**  | Best Investment Opportunity |
# | **Low Attractions + Low Hotels**   | Low demand → Not a target   |
# | **High Hotels + High Attractions** | Competitive, mature markets   |
# | **High Hotels + Low Attractions**  | Oversupply        |
# 


# **Parse location (latitude/longitude)**


# Extract latitude and longitude from the 'Map' column
hotels_df[['Latitude', 'Longitude']] = hotels_df['Map'].str.extract(
    r'([^|]+)\|([^|]+)'   # Capture latitude and longitude separated by |
).astype(float)           # Convert extracted strings to numeric values

# Display first few extracted values
print(hotels_df[['Latitude', 'Longitude']].head())

# **Rating Distribution**


plt.figure(figsize=(6,4))
sns.countplot(x=hotels_df['HotelRating'])
plt.title("Hotel Rating Distribution")
plt.xlabel("Rating (1–5)")
plt.ylabel("Count")
plt.show()

# **Phase 02 - FEATURE ENGINEERING**


# **City Popularity Score**


# Create city popularity scores - More hotels in a city - more popular - more expensive.
city_popularity = hotels_df['cityName'].value_counts().rename('city_popularity')

hotels_df = hotels_df.merge(city_popularity, left_on='cityName', right_index=True)

# **HotelRating to Numeric Format**


# Clean & Convert HotelRating to Numeric Format

# Step 1 - Check all unique values in HotelRating (before conversion)
print("Unique HotelRating values before conversion:")
display(hotels_df['HotelRating'].unique())

# Step 2 - Convert Using Map
rating_map = {
    'FiveStar': 5,
    'FourStar': 4,
    'ThreeStar': 3,
    'TwoStar': 2,
    'OneStar': 1,
    'All': 0
}

# Ensure the column is of string type, strip any whitespace, then map
hotels_df['HotelRating'] = hotels_df['HotelRating'].astype(str).str.strip().map(rating_map)

# Step 3 - Verify Conversion
print("\nUnique HotelRating values after conversion:")
display(hotels_df['HotelRating'].unique())
print("\nValue counts for HotelRating after conversion:")
display(hotels_df['HotelRating'].value_counts())

# **Hotel Facilities Extraction from Text**


# NLP-Based Amenity or Hotel Facilities Extraction Pipeline

import re

# 1. Define standardized amenity categories
amenity_dict = {
    "wifi": [
        "wifi", "free wifi", "internet access", "internet services",
        "wifi available in all areas"
    ],
    "parking": [
        "parking", "free parking", "self parking", "private parking",
        "parking on site"
    ],
    "pool": [
        "pool", "outdoor pool", "indoor pool"
    ],
    "smoke_free": [
        "smoke-free property", "non-smoking", "non-smoking throughout"
    ],
    "pet_friendly": [
        "pets allowed", "pet-friendly"
    ],
    "gym": [
        "fitness facilities", "fitness", "gym"
    ],
    "bar": [
        "bar", "drinking"
    ],
    "restaurant": [
        "restaurant", "dining", "snacking"
    ],
    "air_conditioning": [
        "air conditioning"
    ],
    "multilingual_staff": [
        "multilingual staff", "languages spoken"
    ],
    "breakfast": [
        "breakfast"
    ],
    "kids_services": [
        "for the kids"
    ],
    "transport_access": [
        "getting around"
    ],
    "spa": [
        "spa", "spa services", "massage", "wellness centre", "wellness center"
    ],
    "laundry": [
        "laundry", "laundry service", "dry cleaning", "washing service"
    ]
}

# 2. NLP function to extract amenities from text
def extract_clean_amenities(text):
    """
    Extract clean, standardized amenities from raw hotel facility descriptions.

    Parameters:
        text (str): Raw 'HotelFacilities' text from hotels_df

    Returns:
        list: Cleaned amenity labels such as:
              ['wifi', 'parking', 'smoke_free']
    """
    text = str(text).lower()   # Normalize text
    extracted = []             # Store matched amenity labels

    # Loop through each standardized amenity category
    for clean_label, keywords in amenity_dict.items():
        for keyword in keywords:
            if keyword in text:
                extracted.append(clean_label)
                break  # Stop checking further keywords for this label
    
    return list(set(extracted))  # Remove duplicates

# 3. Apply NLP extraction to hotels_df
hotels_df['clean_amenities'] = hotels_df['HotelFacilities'].apply(extract_clean_amenities) # list of extracted amenities
hotels_df['clean_amenity_count'] = hotels_df['clean_amenities'].apply(len) # number of amenities found

print("\nSample Extracted Amenities:")
print(hotels_df[['HotelFacilities', 'clean_amenities']].head(10))

print("\nTop 20 Extracted Amenities:")
from collections import Counter
all_clean_amenities = Counter([a for sub in hotels_df['clean_amenities'] for a in sub])
print(all_clean_amenities.most_common(20))


import matplotlib.pyplot as plt
from collections import Counter

# Flatten the list of amenities from all hotels
all_amenities = [amenity for sublist in hotels_df['clean_amenities'] for amenity in sublist]

# Count top amenities
amenity_counts = Counter(all_amenities).most_common(10)

# Separate names and values for plotting
amenity_names = [item[0] for item in amenity_counts]
amenity_values = [item[1] for item in amenity_counts]

# Plot
plt.figure(figsize=(12, 6))
plt.barh(amenity_names, amenity_values)
plt.xlabel("Count")
plt.ylabel("Amenity")
plt.title("Top 10 Most Common Hotel Amenities")
plt.gca().invert_yaxis()  # Highest count on top
plt.tight_layout()
plt.show()


# **Hotel Facilities per Record Count**


# Convert Amenities → Numeric without duplicates

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# 1. Ensure list type
hotels_df['clean_amenities'] = hotels_df['clean_amenities'].apply(
    lambda x: x if isinstance(x, list) else []
)

# 2. Amenity count
hotels_df['clean_amenity_count'] = hotels_df['clean_amenities'].apply(len)

# 3. Multi-hot encode amenities
mlb = MultiLabelBinarizer()
amenity_encoded = mlb.fit_transform(hotels_df['clean_amenities'])

amenity_numeric_df = pd.DataFrame(
    amenity_encoded,
    columns=mlb.classes_,
    index=hotels_df.index
)

# 4. Remove any duplicate columns BEFORE merging
existing_cols = set(hotels_df.columns)
new_cols = set(amenity_numeric_df.columns)

duplicate_cols = list(existing_cols.intersection(new_cols))

if duplicate_cols:
    hotels_df.drop(columns=duplicate_cols, inplace=True)

# 5. Now concatenate safely
hotels_df = pd.concat([hotels_df, amenity_numeric_df], axis=1)

print(hotels_df.head())


# Column Information
hotels_df.info()

# **Normalize Engineered Features & Compute Final Location Score**


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create a final composite score
df_final = hotels_df.copy()

# 1️. Select the columns needed for scoring
features = ["HotelRating", "clean_amenity_count", "desc_len", "city_popularity"]

# Replace missing or invalid values (optional but recommended)
df_final[features] = df_final[features].fillna(0)

# 2. Normalize each feature using MinMaxScaler
scaler = MinMaxScaler()
df_final[[f"{col}_norm" for col in features]] = scaler.fit_transform(df_final[features])

# 3️. Apply weights
weights = {
    "HotelRating_norm": 0.40,
    "clean_amenity_count_norm": 0.25,
    "desc_len_norm": 0.20,
    "city_popularity_norm": 0.15
}

# 4️. Compute the weighted composite score
df_final["Final_Composite_Score"] = (
    df_final["HotelRating_norm"]   * weights["HotelRating_norm"] +
    df_final["clean_amenity_count_norm"] * weights["clean_amenity_count_norm"] +
    df_final["desc_len_norm"] * weights["desc_len_norm"] +
    df_final["city_popularity_norm"] * weights["city_popularity_norm"]
)

# 5️. Sort hotels by score (optional)
df_step15 = df_final.sort_values("Final_Composite_Score", ascending=False)

# Display the top 10 highest-scoring hotels
df_final.head(10)


# Top 10 hotels by Composite Score
top10 = df_step15.head(10).copy()

plt.figure(figsize=(12, 6))

# Line plot (NOT bar)
plt.plot(top10["HotelName"], 
         top10["Final_Composite_Score"], 
         marker='o')

plt.xticks(rotation=45, ha='right')
plt.xlabel("Hotel Name")
plt.ylabel("Final Composite Score")
plt.title("Top 10 Highest-Scoring Hotels")
plt.tight_layout()
plt.show()


# **Interpretation**
# 
# 1. The first hotel is clearly the best - (Waldorf Astoria Golf Club) has a higher score than all others.
# - This means it: Has very good ratings, Offers many amenities, Is in a popular city, Has a strong overall profile, It stands out from the rest.
# 
# 2. The next 2–3 hotels are also very strong - Shangri-La and M by Montcalm have scores close to each other.
# - This means these hotels are also top-quality and offer a premium experience.
# 
# 3. After the top 3, scores decrease slowly - The other hotels (positions 5 to 10) have similar quality.
# - This shows: These hotels are still very good, The differences between them are small, They compete very closely with each other
# 
# **Business Insights**
# 
# 1. The highest-scoring hotel is a clear leader
# - It can attract: High-paying guests, Premium marketing partnerships
# 
# 2. Many hotels offer similar quality
# - So the competition is strong, Hotels need to improve amenities or service to stand out.
# 
# 3. Even Rank 7–10 hotels are strong options
# 
# They still have high potential for: Marketing, Premium listings, Higher pricing (if they add a few more amenities)
# 
# 4. The score helps identify the best hotels for business
# 
# Composite score can help: Travel websites promote top hotels, Hotel chains understand where they stand, Investors see which hotels have strong offerings


# **Clustering**


from sklearn.cluster import KMeans

# K-means clustering
df_cluster = df_final.copy()

# Normalized features for clustering
cluster_features = [
    "HotelRating_norm",
    "clean_amenity_count_norm",
    "desc_len_norm",
    "city_popularity_norm"
]

X = df_cluster[cluster_features]

# Run KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Save cluster labels
labels = kmeans.fit_predict(X)

# Add labels to both dataframes
df_cluster["cluster"] = labels
df_final["cluster"] = labels

# Create a summary table with cluster and hotel count
cluster_summary = df_final.groupby("cluster").size().reset_index(name="Hotel_count")

# Sort by cluster number
cluster_summary = cluster_summary.sort_values("cluster").reset_index(drop=True)

print(cluster_summary)


# **Interpretation**
# 
# Hotels have been grouped into 3 clusters based on normalized features like hotel rating, number of amenities, description length, and city popularity.
# 
# - Cluster 0: 363,220 hotels
# - Cluster 1: 302,448 hotels
# - Cluster 2: 344,365 hotels


# Cluster profiling: calculate mean of original features per cluster
cluster_profile = df_final.groupby("cluster")[[
    "HotelRating",
    "clean_amenity_count",
    "desc_len",
    "city_popularity"
]].mean()

# Add count of hotels per cluster
cluster_profile["hotel_count"] = df_final.groupby("cluster").size()

# Add percentage of total hotels as integer
cluster_profile["percentage"] = (cluster_profile["hotel_count"] / df_final.shape[0] * 100).round(0).astype(int)

# Sort by cluster label
cluster_profile = cluster_profile.sort_index()

print(cluster_profile)


# **Interpretation**
# 
# Cluster 0 (36% of hotels, 363,220 hotels):
# - HotelRating: ~2.81 → mid-range hotels
# - Amenities: ~3.32 → fewer amenities
# - Description length: ~173 → moderate detail
# - City popularity: ~493 → located in highly popular cities
# 
# Cluster 1 (30% of hotels, 302,448 hotels):
# - HotelRating: ~3.28 → higher-rated hotels
# - Amenities: ~7.64 → more amenities
# - Description length: ~230 → more detailed descriptions
# - City popularity: ~363 → less popular cities than cluster 0
# 
# Cluster 2 (34% of hotels, 344,365 hotels):
# - HotelRating: ~0.08 → very low-rated hotels
# - Amenities: ~4.45 → moderate amenities
# - Description length: ~167 → shorter descriptions
# - City popularity: ~324 → less popular cities


import seaborn as sns
import matplotlib.pyplot as plt

# Add cluster type manually based on your interpretation
cluster_profile['Cluster_Type'] = ['Mid-range', 'High-rated', 'Low-rated']

# Select features for heatmap
heatmap_data = cluster_profile[[
    "HotelRating", 
    "clean_amenity_count", 
    "desc_len", 
    "city_popularity"
]]

# Create figure
plt.figure(figsize=(10, 5))

# Draw heatmap
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

# Add cluster type and percentage as x-axis labels
for idx, (ctype, pct) in enumerate(zip(cluster_profile['Cluster_Type'], cluster_profile['percentage'])):
    plt.text(-0.5, idx + 0.5, f"{ctype} ({pct}%)", ha='right', va='center', fontsize=10, weight='bold')

plt.title("Cluster Profiling Heatmap")
plt.ylabel("")
plt.xlabel("")
plt.show()


# Add Cluster Labels Into the Final Dataset

# 1. Use the fitted KMeans model to generate labels
labels = kmeans.labels_

# 2. Add labels to df_final
df_final['cluster'] = labels

# 3. (Optional) Add labels to df_cluster as well
df_cluster['cluster'] = labels

# 4. Check if added correctly
df_final[['HotelName', 'HotelRating', 'cluster']].head()


df_final.info()

# | Cluster | Star Rating | Rating   | Amenities | Description Length | City Popularity | Type                          |
# | ------- | ----------- | -------- | --------- | ------------------ | --------------- | ----------------------------- |
# | **0**   | 2★ – 3★     | Medium   | Low       | Medium             | High            | Mid-range / Popular City      |
# | **1**   | 3★ – 4★     | High     | High      | Long               | Medium          | High-quality / Many Amenities |
# | **2**   | 0★ – 1★     | Very Low | Medium    | Short              | Low             | Low-quality / Budget          |
# 


# **Model Training**



# RandomForestRegressor for HotelRating
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# 1. SELECT FEATURES - Amenity one-hot columns (boolean 0/1)
amenity_cols = [
    'wifi','parking','pool','spa','restaurant','gym','bar','breakfast',
    'air_conditioning','multilingual_staff','pet_friendly','laundry',
    'kids_services','transport_access','smoke_free'
]

# Numerical engineered features
base_features = [
    'clean_amenity_count',
    'desc_len',
    'city_popularity',
    'Latitude',
    'Longitude'
]

# Combine (only those that exist in the dataset)
feature_cols = [c for c in base_features + amenity_cols if c in df_final.columns]

print("Using features:", feature_cols)


# 2. PREPARE MODEL DATAFRAME
df_model = df_final[feature_cols + ['HotelRating', 'cityName']].copy()

# Impute missing coordinates by city mean
for coord in ['Latitude', 'Longitude']:
    df_model[coord] = df_model.groupby('cityName')[coord].transform(
        lambda g: g.fillna(g.mean())
    )
    df_model[coord].fillna(df_model[coord].median(), inplace=True)

# Drop cityName (not required for model)
df_model.drop(columns=['cityName'], inplace=True)

# Drop any remaining missing values
df_model = df_model.dropna().reset_index(drop=True)

print("Training rows:", df_model.shape[0])


# 3. TRAIN-TEST SPLIT
X = df_model[feature_cols]
y = df_model['HotelRating'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)


# 4. TRAIN RANDOM FOREST MODEL
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


## 5. EVALUATION
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# your sklearn version does NOT support squared=False → compute RMSE manually
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5   # √MSE

r2 = r2_score(y_test, y_pred)

print("\n Random Forest Performance")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Cross-validation
cv_mae = -cross_val_score(
    rf, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
)

print("\n CV MAE (3-fold):", round(cv_mae.mean(), 4))


# 6. FEATURE IMPORTANCE
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\n Top Feature Importances:")
print(importances_sorted.head(20))


# 7. SAVE MODEL
joblib.dump(rf, "RandomForest_HotelRating_Model.joblib")
print("\n Model saved: RandomForest_HotelRating_Model.joblib")


# 8. SAMPLE PREDICTIONS
results = X_test.copy()
results['Actual_Rating'] = y_test.values
results['Predicted_Rating'] = y_pred
results.head(10)

# **Interpretation**
# 
# - Model Performance
# 
# - MAE ~0.85 → average prediction error is less than 1 star
# - RMSE ~1.11 → 1-star error on average
# - R² ~0.469 → model explains ~47% variance
# - CV MAE ~1.06 → generalized performance (slightly lower)
# 
# Reasonable because:
# 1. HotelRating is integer 1–5
# 2. Very noisy text-based / amenity-based predictions
# 3. No price, no user reviews → heavy missing signal
# 
# - Feature Importance Interpretation
# 
# | Feature                             | Influence                     |
# | ----------------------------------- | ----------------------------- |
# | **Longitude**                       | Very high                     |
# | **Latitude**                        | Very high                     |
# | **Spa**                             | Strong indicator of luxury    |
# | **Description Length**              | High                          |
# | **Bar, AC, Pet Friendly**           | Medium                        |
# | **City Popularity**                 | Medium                        |
# | **Amenity Count**                   | Medium                        |
# | **Kids Services, Transport Access** | Almost zero                   |
# 
# 
# - Why are Latitude/Longitude so important?
# Geography strongly correlates with:
# 1. Country → price level → luxury rating
# 2. Resort locations → higher ratings
# 3. Rural areas → lower ratings
# This is normal.


# **HOTEL RATING PREDICTOR**


# HOTEL RATING PREDICTOR — RANDOM FOREST

# Import libraries
import pandas as pd
import numpy as np
from joblib import load

# Load the trained RandomForest model
rf = load("RandomForest_HotelRating_Model.joblib")

# Define required feature columns
feature_columns = [
    'clean_amenity_count', 'desc_len', 'city_popularity',
    'Latitude', 'Longitude',
    'wifi', 'parking', 'pool', 'spa', 'restaurant', 'gym', 'bar',
    'breakfast', 'air_conditioning', 'multilingual_staff', 'pet_friendly',
    'laundry', 'kids_services', 'transport_access', 'smoke_free'
]

# Define the predictor function
def predict_hotel_rating(
    clean_amenity_count,
    desc_len,
    city_popularity,
    Latitude,
    Longitude,
    wifi=0, parking=0, pool=0, spa=0, restaurant=0, gym=0, bar=0,
    breakfast=0, air_conditioning=0, multilingual_staff=0, pet_friendly=0,
    laundry=0, kids_services=0, transport_access=0, smoke_free=0
):
    """
    Predict Hotel Rating (0-5) using RandomForest model.

    Parameters:
    -----------
    clean_amenity_count : int
        Total number of amenities
    desc_len : int
        Description length (number of characters)
    city_popularity : int
        Popularity of the city (hotel count or ranking)
    Latitude, Longitude : float
        Coordinates of hotel location
    All other parameters : 0/1 (binary amenities)
    
    Returns:
    --------
    float : Predicted Hotel Rating (0-5 stars)
    """
    # Row dictionary
    row = {
        'clean_amenity_count': clean_amenity_count,
        'desc_len': desc_len,
        'city_popularity': city_popularity,
        'Latitude': Latitude,
        'Longitude': Longitude,
        'wifi': wifi,
        'parking': parking,
        'pool': pool,
        'spa': spa,
        'restaurant': restaurant,
        'gym': gym,
        'bar': bar,
        'breakfast': breakfast,
        'air_conditioning': air_conditioning,
        'multilingual_staff': multilingual_staff,
        'pet_friendly': pet_friendly,
        'laundry': laundry,
        'kids_services': kids_services,
        'transport_access': transport_access,
        'smoke_free': smoke_free
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([row])

    # Predict
    prediction = rf.predict(df_input)[0]

    # Clip to 0–5 range
    prediction = max(0, min(5, prediction))

    return round(prediction, 2)


# EXAMPLE

# Example 1 — Mid-range hotel
rating1 = predict_hotel_rating(
    clean_amenity_count=12,
    desc_len=450,
    city_popularity=1200,
    Latitude=45.890,
    Longitude=-74.0060,
    wifi=1, parking=1, restaurant=1, breakfast=1, smoke_free=1
)
print("Example 1 Predicted Rating:", rating1)

# Example 2 — Luxury 5-star hotel
rating2 = predict_hotel_rating(
    clean_amenity_count=18,
    desc_len=2000,
    city_popularity=5000,
    Latitude=25.2048,
    Longitude=55.2708,
    wifi=1, pool=1, spa=1, gym=1, bar=1, restaurant=1,
    air_conditioning=1, multilingual_staff=1, pet_friendly=1
)
print("Example 2 Predicted Rating:", rating2)

# Example 3 — Budget Hotel
rating3 = predict_hotel_rating(
    clean_amenity_count=3,
    desc_len=100,
    city_popularity=100,
    Latitude=23.5,
    Longitude=77.4,
    wifi=1
)
print("Example 3 Predicted Rating:", rating3)


# **Dashboard-ready summary dataset**


# Dashboard-ready summary dataset

# Columns to keep
dashboard_columns = [
    "HotelCode", "HotelName", "countyName", "cityName", "Address", "HotelWebsiteUrl",
    "HotelRating", "clean_amenity_count", "desc_len", "city_popularity",
    "Final_Composite_Score", "cluster",
    "Latitude", "Longitude"
]

# Add all binary amenity columns
amenity_columns = [
    'wifi', 'parking', 'pool', 'spa', 'restaurant', 'gym', 'bar',
    'breakfast', 'air_conditioning', 'multilingual_staff', 'pet_friendly',
    'laundry', 'kids_services', 'transport_access', 'smoke_free'
]

dashboard_columns.extend(amenity_columns)

# Add normalized columns
normalized_columns = [
    "HotelRating_norm", "clean_amenity_count_norm", "desc_len_norm", "city_popularity_norm"
]
dashboard_columns.extend(normalized_columns)

# Create the dashboard-ready dataset
dashboard_df = df_final[dashboard_columns].copy()

# Reset index
dashboard_df.reset_index(drop=True, inplace=True)

# Show sample
dashboard_df.info()


# **Business Focused Insights** 


# **1. Which Countries show the highest hotels rates per star category?**


# Group by country and HotelRating, get mean composite score
highest_score_per_star = (
    dashboard_df.groupby(["countyName", "HotelRating"])["Final_Composite_Score"]
    .mean()
    .reset_index()
    .round({"Final_Composite_Score": 2})
)

# Get country with highest score per star
highest_score_per_star = highest_score_per_star.loc[
    highest_score_per_star.groupby("HotelRating")["Final_Composite_Score"].idxmax()
].reset_index(drop=True)

# Rename for clarity
highest_score_per_star.rename(columns={"countyName":"Country", "Final_Composite_Score":"Avg_Score"}, inplace=True)

# Visualization
plt.figure(figsize=(10,6))
sns.barplot(
    data=highest_score_per_star,
    x="HotelRating",
    y="Avg_Score",
    hue="Country",
    dodge=False
)
plt.title("Highest Hotel Value (Final Composite Score) per Star Category by Country")
plt.xlabel("Hotel Star Rating")
plt.ylabel("Average Composite Score")
plt.legend(title="Country")
plt.show()


# **Interpretation** 
# 
# - Final Composite Score” as a measure of hotel value:
# 
# Star Category 0: Qatar has the highest average composite score → hotels in Qatar in this category are likely the most valuable (or highest “rate” proxy) among 0-star hotels.
# 
# Star Category 5: Indonesia has the highest average composite score → 5-star hotels in Indonesia are likely the most valuable (or highest “rate” proxy) among 5-star hotels.


# **2. What Hotel attributes (amenities, review count, rating, location) most influence pricing?**


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Features and target
features = ['HotelRating', 'clean_amenity_count', 'city_popularity', 'desc_len']
X = dashboard_df[features]
y = dashboard_df['Final_Composite_Score']

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values()

# Horizontal bar chart
plt.figure(figsize=(10,6))
bars = plt.barh(importance.index, importance.values, color=['#66b3ff','#ff9999','#99ff99','#ffcc99'])
plt.xlabel('Importance')
plt.title('Feature Importance for Hotel Pricing (Proxy)')

# Add % labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width*100:.1f}%', va='center')

plt.xlim(0, max(importance.values)*1.2)
plt.show()


# | Feature                 | Importance | Interpretation                                                                                                        |
# | ----------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------- |
# | **HotelRating**         | 0.8377     | This dominates the “pricing” signal. Hotels with higher ratings strongly correlate with higher Final_Composite_Score. |
# | **clean_amenity_count** | 0.1361     | Amenities like pools, gyms, breakfast, etc., moderately influence the score.                                          |
# | **city_popularity**     | 0.0173     | Minimal effect. Popular cities slightly raise the score, but not significant.                                         |
# | **desc_len**            | 0.0088     | Hotel description length has almost no effect.                                                                        |
# 


# **3. How can hotel chains/OTA's optimise partner portfolio by identifying underserved segments (example: budget hotels in permium city zones) or over-saturated markets?** 


# **A. Identify Underserved Segments - High demand but low supply**
# 
# - Imagine a premium city zone like downtown New York
# - Most hotels there are luxury hotels, with very high prices 
# - Budget travelers struggle to find affordable options
# 
# Opportunity: A budget hotel chain or OTA can partner with budget hotels in this zone, because there is demand but not enough supply. This is an underserved segment.


# **B. Identify Over Saturated Segments - High supply, high competition**
# 
# - Imagine a suburb with many mid-range hotels, all competing heavily for the same customers.
# - Adding more hotels there might not increase revenue, because the market is already crowded.
# 
# Action: The hotel chain/OTA might avoid adding new partners here or use discounts/marketing to stand out, because it’s an over-saturated market.


# **4. Which features [Hotel Facilities] drive pricing for clusters 0 [Mid Range], 1 [High Rated], and 2 [Low Rated]**
# 


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Step 1: Filter Cluster 0
cluster_0_df = dashboard_df[dashboard_df['cluster'] == 0]

# Step 2: Select Features and Target
features = [ 'wifi', 'parking', 'pool', 'spa',
             'gym', 'bar', 'breakfast', 'air_conditioning', 'multilingual_staff',
             'pet_friendly', 'laundry', 'kids_services', 'transport_access', 'smoke_free']

X = cluster_0_df[features]
y = cluster_0_df['Final_Composite_Score']

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 4: Extract Feature Importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# Step 5: Heatmap with Percentage Annotation
plt.figure(figsize=(10,6))
sns.heatmap(importance.to_frame() * 100, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label':'Importance (%)'})
plt.title('Amenity Impact on Pricing (Cluster 0 - Mid Range)', fontsize=14)
plt.ylabel('Feature', fontsize=12)
plt.xlabel('')
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Step 1: Filter Cluster 1
cluster_1_df = dashboard_df[dashboard_df['cluster'] == 1]

# Step 2: Select Features and Target
features = [ 'wifi', 'parking', 'pool', 'spa',
             'gym', 'bar', 'breakfast', 'air_conditioning', 'multilingual_staff',
             'pet_friendly', 'laundry', 'kids_services', 'transport_access', 'smoke_free']

X = cluster_1_df[features]
y = cluster_1_df['Final_Composite_Score']

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 4: Extract Feature Importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# Step 5: Heatmap with Percentage Annotation
plt.figure(figsize=(10,6))
sns.heatmap(importance.to_frame() * 100, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label':'Importance (%)'})
plt.title('Amenity Impact on Pricing (Cluster 1 - High Range)', fontsize=14)
plt.ylabel('Feature', fontsize=12)
plt.xlabel('')
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Step 1: Filter Cluster 2
cluster_2_df = dashboard_df[dashboard_df['cluster'] == 2]

# Step 2: Select Features and Target
features = [ 'wifi', 'parking', 'pool', 'spa',
             'gym', 'bar', 'breakfast', 'air_conditioning', 'multilingual_staff',
             'pet_friendly', 'laundry', 'kids_services', 'transport_access', 'smoke_free']

X = cluster_0_df[features]
y = cluster_0_df['Final_Composite_Score']

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 4: Extract Feature Importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# Step 5: Heatmap with Percentage Annotation
plt.figure(figsize=(10,6))
sns.heatmap(importance.to_frame() * 100, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label':'Importance (%)'})
plt.title('Amenity Impact on Pricing (Cluster 2 - Low Range)', fontsize=14)
plt.ylabel('Feature', fontsize=12)
plt.xlabel('')
plt.tight_layout()
plt.show()