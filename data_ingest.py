# data_ingest.py
import pandas as pd
import re

def load_and_clean(file_path):
    df = pd.read_csv(file_path)
    
    # Drop rows with missing text
    df.dropna(subset=['Text'], inplace=True)
    
    # Clean text
    df['text'] = df['Text'].apply(lambda x: re.sub(r'<.*?>', '', str(x)).strip())
    
    # Create metadata column
    df['metadata'] = df['Summary'].fillna('') + " | " + df['ProductId'].astype(str)
    print(df.head())
    
    # Keep only required columns
    cleaned_df = df[['Id', 'text', 'metadata']].rename(columns={'Id': 'id'})
    
    return cleaned_df

if __name__ == "__main__":
    df = load_and_clean("/home/divyaks1242/Documents/Rag/amaon_review/cleaned_amazon_review.csv")
    df.to_csv("cleaned_dataset.csv", index=False)
    print(df.head())

