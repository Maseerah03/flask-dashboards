from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import io
import base64

# Download NLP resources
nltk.download("vader_lexicon")
import spacy.cli

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# ?? Function to analyze sentiment
def get_sentiment(review):
    score = sia.polarity_scores(review)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ?? Function to generate trend graph
def generate_trend_graph(product_name, aspect):
    try:
        df = pd.read_csv("review_trends.csv")

        if "Timestamp" not in df.columns:
            return None, "Timestamp column missing in dataset."

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df_filtered = df[df["Product Name"].str.contains(product_name, case=False, na=False)]

        if df_filtered.empty:
            return None, f"No reviews found for '{product_name}'."

        df_filtered = df_filtered[df_filtered["Review"].str.contains(aspect, case=False, na=False)]
        if df_filtered.empty:
            return None, f"No reviews found for '{product_name}' and aspect '{aspect}'."

        df_filtered["Sentiment"] = df_filtered["Review"].apply(get_sentiment)
        df_filtered["Month-Year"] = df_filtered["Timestamp"].dt.to_period("M")

        sentiment_trends = df_filtered.groupby(["Month-Year", "Sentiment"]).size().unstack(fill_value=0)

        # Plot Graph
        plt.figure(figsize=(10, 5))
        sentiment_trends.plot(kind="line", marker="o", colormap="viridis")
        plt.xlabel("Month-Year")
        plt.ylabel("Number of Reviews")
        plt.title(f"?? '{aspect}' Sentiment Trends for '{product_name}' Over Time")
        plt.legend(title="Sentiment", loc="upper left")
        plt.xticks(rotation=45)
        plt.grid(True)

        # Convert Graph to Image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url, None

    except FileNotFoundError:
        return None, "Dataset not found! Please generate 'review_trends.csv'."

# ?? Route for **Product Recommendation Dashboard**
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        product_name = request.form.get("product", "").strip()
        aspect = request.form.get("aspect", "").strip()

        if not product_name or not aspect:
            return render_template("index.html", plot_url=None, error="? Please enter both product and aspect.")

        plot_url, error = generate_trend_graph(product_name, aspect)
        return render_template("index.html", plot_url=plot_url, error=error)

    return render_template("index.html", plot_url=None, error=None)

# ?? Route for **Manufacturer Dashboard**
@app.route("/manufacturer", methods=["GET", "POST"])
def manufacturer_dashboard():
    if request.method == "POST":
        product_name = request.form.get("product", "").strip()
        aspect = request.form.get("aspect", "").strip()

        if not product_name or not aspect:
            return render_template("manufacturer_dashboard.html", plot_url=None, error="? Please enter both product and aspect.")

        plot_url, error = generate_trend_graph(product_name, aspect)
        return render_template("manufacturer_dashboard.html", plot_url=plot_url, error=error)

    return render_template("manufacturer_dashboard.html", plot_url=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
