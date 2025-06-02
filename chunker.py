# chunker.py
def chunk_text(text, doc_id, size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = ' '.join(words[i:i+size])
        chunks.append((doc_id, i // size, chunk))
    return chunks

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("cleaned_dataset.csv")
    
    all_chunks = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['text'], row['id'])
        for doc_id, chunk_id, text in chunks:
            all_chunks.append({'doc_id': doc_id, 'chunk_id': chunk_id, 'text': text, 'metadata': row['metadata']})
    
    chunk_df = pd.DataFrame(all_chunks)
    chunk_df.to_csv("chunked_dataset.csv", index=False)
    print(chunk_df.head())
