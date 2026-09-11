import pandas as pd

# Load the dataset directly from GitHub
url = "https://raw.githubusercontent.com/Vonter/india-mplads-works/main/csv/MPLADS.csv"

df = pd.read_csv(
    url,
    sep=";",              # Important: the file uses semicolon as separator
    low_memory=False,     # Helps with large files
    encoding="utf-8"
)

# Quick check
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())