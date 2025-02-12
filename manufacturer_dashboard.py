from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import io
import base64

# Download NLP resources (if not already installed)
nltk.download("vader_lexicon")
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Function to analyze sentiment
def get_sentiment(review):
    score = sia.polarity_scores(review)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to generate trend graph for a specific model or product ID
def generate_trend_graph(product_model, aspect):
    try:
        df = pd.read_csv("review_trends.csv")

        # Ensure required columns exist
        if "Timestamp" not in df.columns or "Product Name" not in df.columns:
            return None, "Dataset missing required columns (Timestamp, Product Name)."

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # ?? **Filter by Specific Product Model OR Product ID**
        df_filtered = df[df["Product Name"].str.contains(product_model, case=False, na=False)]

        if df_filtered.empty:
            return None, f"No reviews found for Product '{product_model}'."

        # ?? Further filter by Aspect (e.g., "Camera", "Battery")
        df_filtered = df_filtered[df_filtered["Review"].str.contains(aspect, case=False, na=False)]

        if df_filtered.empty:
            return None, f"No reviews found for '{product_model}' and aspect '{aspect}'."

        # Sentiment Analysis
        df_filtered["Sentiment"] = df_filtered["Review"].apply(get_sentiment)
        df_filtered["Month-Year"] = df_filtered["Timestamp"].dt.to_period("M")

        # Group by Sentiment Over Time
        sentiment_trends = df_filtered.groupby(["Month-Year", "Sentiment"]).size().unstack(fill_value=0)

        # ?? **Plot Graph**
        plt.figure(figsize=(10, 5))
        sentiment_trends.plot(kind="line", marker="o", colormap="viridis")
        plt.xlabel("Month-Year")
        plt.ylabel("Number of Reviews")
        plt.title(f" {product_model} {aspect} Sentiment Trends ")
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

# Function for Refund Optimization (Fixed for Specific Models)
def get_refund_optimization(product_model):
    try:
        df = pd.read_csv("review_trends.csv")

        # Ensure required columns exist
        if "Timestamp" not in df.columns or "Product Name" not in df.columns:
            return None, "Dataset missing required columns (Timestamp, Product Name)."

        df_filtered = df[df["Product Name"].str.contains(product_model, case=False, na=False)]

        if df_filtered.empty:
            return None, f"No reviews found for '{product_model}'."

        # Filter negative reviews
        df_filtered["Sentiment"] = df_filtered["Review"].apply(get_sentiment)
        df_negative = df_filtered[df_filtered["Sentiment"] == "Negative"]

        if df_negative.empty:
            return None, f"No negative reviews found for '{product_model}'."

        # **? Remove Duplicate Reviews**
        df_negative = df_negative.drop_duplicates(subset=["Review"])

        # Get top 5 most recent negative reviews for refund analysis
        refund_data = df_negative[["Timestamp", "Review"]].sort_values(by="Timestamp", ascending=False).head(5)

        return refund_data.to_html(classes="table table-striped table-dark", index=False), None

    except FileNotFoundError:
        return None, "Dataset not found! Please generate 'review_trends.csv'."

# Flask Route for Manufacturer Dashboard
@app.route("/", methods=["GET", "POST"])
def index():
    trend_plot = None
    refund_data = None
    error = None

    if request.method == "POST":
        product_model = request.form.get("product_model", "").strip()
        aspect = request.form.get("aspect", "").strip()
        action = request.form.get("action")

        if not product_model:
            return render_template("manufacturer_dashboard.html", trend_plot=None, refund_data=None, error="? Please enter a Product Model or ID.")

        if action == "track_trends":
            if not aspect:
                return render_template("manufacturer_dashboard.html", trend_plot=None, refund_data=None, error="? Please enter an Aspect for Trend Tracking.")
            trend_plot, error = generate_trend_graph(product_model, aspect)

        elif action == "refund_optimization":
            refund_data, error = get_refund_optimization(product_model)

    return render_template("manufacturer_dashboard.html", trend_plot=trend_plot, refund_data=refund_data, error=error)

if __name__ == "__main__":
    app.run(debug=True)
