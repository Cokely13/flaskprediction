import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("training_data.csv")

# Quick look at the data
print("First 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(data.describe())

# Plot distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(data['rating'], bins=10, kde=True, color='blue')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Genre Analysis
print("\nTop Genres:")
data['genres'] = data['genres'].fillna("Unknown")  # Handle missing genres
all_genres = data['genres'].str.split('|').explode()  # Split genres and count
top_genres = all_genres.value_counts()
print(top_genres)

# Plot top genres
plt.figure(figsize=(10, 6))
top_genres.head(10).plot(kind='bar', color='green')
plt.title("Top Genres")
plt.xlabel("Genre")
plt.ylabel("Frequency")
plt.show()
