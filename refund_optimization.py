import pandas as pd

# Load review trends CSV
df = pd.read_csv("review_trends.csv")

# Filter negative reviews
negative_reviews = df[df["Review"].str.contains("bad|worst|poor|refund|return|broke|defective", case=False, na=False)]

# Save to a new CSV file
negative_reviews.to_csv("refund_optimization.csv", index=False)

print("? Refund optimization data saved as refund_optimization.csv") 
