from flask import Flask, render_template, request
import requests
import random
from bs4 import BeautifulSoup
from collections import defaultdict
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import os
from datetime import datetime

# Download NLP resources
nltk.download("vader_lexicon")
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Scraper API Key (Replace with your key)
SCRAPER_API_KEY = "38cacfcc88819fcce10762c3826a6b94"

# User-Agent List (to avoid blocking)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

# Function to get product links from Amazon
def get_product_links(product_name, max_products=5):
    search_url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
    api_url = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url={search_url}"
    
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    product_elements = soup.select("div.s-main-slot div[data-asin] a.a-link-normal.s-no-outline")

    product_links = []
    for element in product_elements[:max_products]:
        link = element.get("href")
        if link and link.startswith("/"):
            product_links.append("https://www.amazon.in" + link)
    
    return product_links

# Function to extract **actual product name** from reviews
def extract_product_name(review_text, default_name="Unknown Product"):
    doc = nlp(review_text)
    product_name = []

    for token in doc:
        if token.ent_type_ in ["PRODUCT", "ORG"]:  # Check for product-related named entities
            product_name.append(token.text)

    if product_name:
        return " ".join(product_name)
    return default_name  # If no product name found, return "Unknown Product"

# Function to scrape reviews & timestamps
def scrape_reviews(product_url, num_pages=3):
    reviews_data = []

    for page in range(1, num_pages + 1):
        review_url = f"{product_url}/ref=cm_cr_arp_d_paging_btm_next_{page}?pageNumber={page}"
        api_url = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url={review_url}"

        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.find_all("span", {"data-hook": "review-body"})
        date_elements = soup.find_all("span", {"data-hook": "review-date"})

        for review, date in zip(review_elements, date_elements):
            review_text = review.text.strip().replace("Read more", "").strip()
            review_date = date.text.strip().replace("Reviewed in India on", "").strip()

            product_name = extract_product_name(review_text, default_name="Unknown Product")  # ? Extract actual product name
            reviews_data.append({"Review": review_text, "Timestamp": review_date, "Product Name": product_name})

    return reviews_data  # ? Return correct product name instead of just "MOBILE"

# Function to save reviews for trend tracking
def save_reviews_to_csv(reviews_data):
    df = pd.DataFrame(reviews_data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")  

    file_exists = os.path.exists("review_trends.csv")
    df.to_csv("review_trends.csv", mode="a", index=False, header=not file_exists)

# Function to analyze sentiment of aspects
def analyze_aspect_sentiment(reviews):
    aspect_sentiments = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})

    for review in reviews:
        doc = nlp(review["Review"])
        sentiment_score = sia.polarity_scores(review["Review"])["compound"]

        for token in doc:
            if token.pos_ == "NOUN":
                aspect = token.text.lower()
                if sentiment_score > 0.05:
                    aspect_sentiments[aspect]["Positive"] += 1
                elif sentiment_score < -0.05:
                    aspect_sentiments[aspect]["Negative"] += 1
                else:
                    aspect_sentiments[aspect]["Neutral"] += 1

    return aspect_sentiments

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        product_name = request.form["product"].strip()
        chosen_aspect = request.form["aspect"].strip().lower()

        if not product_name or not chosen_aspect:
            return render_template("index.html", links=[], error="Please enter both product and aspect.")

        product_links = get_product_links(product_name)

        if not product_links:
            return render_template("index.html", links=[], error="No products found! Try another search.")

        product_reviews = []
        for product_url in product_links:
            reviews = scrape_reviews(product_url)
            if reviews:
                product_reviews.extend(reviews)  # Store all reviews correctly
                save_reviews_to_csv(reviews)  # ? Save actual product names

        return render_template("index.html", links=product_links, error=None)

    return render_template("index.html", links=[], error=None)

if __name__ == "__main__":
    app.run(debug=True)
