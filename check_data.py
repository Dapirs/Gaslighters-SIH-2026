import pandas as pd

url = "https://raw.githubusercontent.com/Vonter/india-mplads-works/main/csv/MPLADS.csv"
df = pd.read_csv(url, sep=";", low_memory=False, encoding="utf-8")

print("DTYPES:")
print(df.dtypes)

print("\nNULL COUNTS:")
print(df.isnull().sum())

print("\nSTATUS VALUES:")
print(df["STATUS"].value_counts())

print("\nALLOCATION AMOUNT sample:")
print(df["ALLOCATION AMOUNT"].head(10))

print("\nCATEGORY VALUES:")
print(df["CATEGORY"].value_counts().head(10))

print("\nIDA APPROVAL VALUES:")
print(df["IDA APPROVAL"].value_counts())

# ── Added checks ──────────────────────────────────────────────
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nALLOCATION AMOUNT <= 0 (breaks Benford's Law + is_round_amount):")
print((df["ALLOCATION AMOUNT"] <= 0).sum())

print("\nMISSING LOCATION FIELDS (feeds missing_location flag):")
for col in ["CITY", "WARD", "VILLAGE"]:
    if col in df.columns:
        print(f"  {col}: {df[col].isnull().sum()} missing ({df[col].isnull().mean()*100:.1f}%)")