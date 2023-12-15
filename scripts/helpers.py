import pandas as pd


def read_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
