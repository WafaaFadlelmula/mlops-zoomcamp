import pandas as pd

# Quick test
print("Testing output...")

# Load the actual data for Question 3
url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
df = pd.read_parquet(url)

print(f"QUESTION 3 ANSWER: {len(df)} records")