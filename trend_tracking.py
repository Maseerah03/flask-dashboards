import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime

# Download NLP model if not already installed
nltk.download("vader_lexicon")

# Load NLP model & sentiment analyzer
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Load the review trends dataset
csv_file = "review_trends.csv"  # File generated from app.py
df = pd.read_csv(csv_file)

# Convert review timestamps to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# User input for aspect tracking
selected_aspect = input("Enter the aspect you want to track (e.g., camera, battery, price): ").strip().lower()

# Filter reviews that mention the selected aspect
filtered_reviews = df[df["Review"].str.contains(selected_aspect, case=False, na=False)].copy()

if filtered_reviews.empty:
    print(f"No reviews found for aspect '{selected_aspect}'. Try another aspect.")
    exit()

# Sentiment classification function
def get_sentiment(review):
    score = sia.polarity_scores(review)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis (Fix: Using .loc to avoid SettingWithCopyWarning)
filtered_reviews.loc[:, "Sentiment"] = filtered_reviews["Review"].apply(get_sentiment)

# Count sentiment over time
sentiment_trends = filtered_reviews.groupby([filtered_reviews["Timestamp"].dt.to_period("M"), "Sentiment"]).size().unstack(fill_value=0)

# ?? **Plot Sentiment Trend Graph**
plt.figure(figsize=(12, 6))

# Fix: If only 1 unique month, use bar chart instead of line graph
if len(sentiment_trends.index) == 1:
    sentiment_trends.plot(kind="bar", colormap="viridis")
else:
    sentiment_trends.plot(kind="line", marker="o", colormap="viridis")

plt.xlabel("Month-Year")
plt.ylabel("Number of Reviews")
plt.title(f"Sentiment Trend for '{selected_aspect}' Aspect Over Time")
plt.legend(title="Sentiment")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
