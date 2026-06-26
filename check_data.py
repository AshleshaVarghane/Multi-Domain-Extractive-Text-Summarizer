import pandas as pd
import sys

CSV_PATH = "data/global_news/data.csv"

def main():
    print(f"Inspecting: {CSV_PATH}")
    try:
        cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
        print("Columns:", cols)
    except Exception as e:
        print("Failed to read CSV header:", e, file=sys.stderr)

    try:
        sample = pd.read_csv(CSV_PATH, nrows=5)
        print("\nFirst 5 rows:")
        print(sample.head().to_string(index=False))
    except Exception as e:
        print("Failed to read sample rows:", e, file=sys.stderr)
 
        try:
            it = pd.read_csv(CSV_PATH, chunksize=10000)
            first = next(it)
            print("\nRead first chunk columns:", first.columns.tolist())
            print("\nFirst chunk sample:")
            print(first.head().to_string(index=False))
        except Exception as e2:
            print("Chunked read also failed:", e2, file=sys.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
