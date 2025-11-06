import time

import fitz
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rake_nltk import Rake
import requests
import numpy as np
import json

# PyMuPDF Open the PDF file
pdf_document = "document.pdf"
document = fitz.open(pdf_document)
# Initialize a dictionary to hold the text for each page
pdf_text = {}
# Loop through each page
for page_number in range(document.page_count):
# Get a page
    page = document.load_page(page_number)
    # Extract text from the page
    text = page.get_text()
    # Store the extracted text in the dictionary
    pdf_text[page_number + 1] = text
# Pages are 1-indexed for readability
# Close the document
document.close()
# Output the dictionary
for page, text in pdf_text.items():
    print(f"Text from page {page}:\n{text}\n")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split text into chunks
    page_chunks = {}
    for page, text in pdf_text.items():
        chunks = text_splitter.split_text(text)
        page_chunks[page] = chunks
    # Output chunks for each page
    for page, chunks in page_chunks.items():
        print(f"Text chunks from page {page}:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"Chunk {i}:\n{chunk}\n")

rake = Rake()

# Extract phrases from each page and store in a dictionary
page_phrases = {}
for page, text in pdf_text.items():
    rake.extract_keywords_from_text(text)
    phrases = rake.get_ranked_phrases()
    page_phrases[page] = phrases

chunk_phrases = {}
# Extract phrases for each chunk
for page, chunks in page_chunks.items():
    for chunk_number, chunk in enumerate(chunks, start=1):
        rake.extract_keywords_from_text(chunk)
        phrases = rake.get_ranked_phrases()
        chunk_phrases[(page, chunk_number)] = phrases
# Output phrases for each chunk
for (page, chunk_number), phrases in chunk_phrases.items():
    print(f"Key phrases from page {page}, chunk {chunk_number}:\n{phrases}\n")

def get_embedding(phrase):
    """Получение эмбеддингов через Ollama API"""

    # Ollama использует отдельный endpoint для эмбеддингов
    url = "http://localhost:11434/api/embed"

    data = {
        "model": "nomic-embed-text",
        "prompt": phrase
    }

    try:
        responsee = requests.post(url, json=data)
        responsee.raise_for_status()

        result = responsee.json()
        return result['embedding']

    except requests.exceptions.RequestException as e:
        return None
    except KeyError:
        return None
# Dictionary to hold embeddings
phrase_embeddings = {}
# Generate embeddings for each phrase
for (page, chunk_number), phrases in chunk_phrases.items():
    embeddings = [get_embedding(phrase) for phrase in phrases]
    phrase_embeddings[(page, chunk_number)] = list(zip(phrases, embeddings))
# Prepare data for Excel
excel_data = []
for (page, chunk_number), phrases in phrase_embeddings.items():
    for phrase, embedding in phrases:
        excel_data.append({ "Page": page, "Chunk": chunk_number, "Phrase": phrase, "Embedding": embedding })
# Create a DataFrame
df = pd.DataFrame(excel_data)
# Save to Excel
excel_filename = "phrases_embeddings.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Embeddings saved to {excel_filename}")

def extract_phrases_from_query(query):
    rake.extract_keywords_from_text(query)
    return rake.get_ranked_phrases()
# Example query
query = "Антон?"
# Extract phrases from the query
query_phrases = extract_phrases_from_query(query)
# Output query phrases
print(f"Query phrases:\n{query_phrases}\n")


def get_embeddings(phrases):
    """Получение эмбеддингов для списка фраз через Ollama"""
    embeddingss = []

    for phrase in phrases:
        url = "http://localhost:11434/api/embeddings"
        data = {
            "model": "nomic-embed-text",
            "prompt": phrase
        }

        try:
            responsee = requests.post(url, json=data)
            responsee.raise_for_status()
            result = responsee.json()
            embeddingss.append(result['embedding'])
            time.sleep(1)
        except Exception as e:
            print(f"Ошибка для фразы '{phrase}': {e}")
            embeddingss.append(None)

    return embeddingss
# Get embeddings for query phrases
query_embeddings = get_embeddings(query_phrases)

import numpy as np
from scipy.spatial.distance import cosine
# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    print("vec1: " + vec1)
    print("vec2: " + vec2)
    if vec1 is None or vec2 is None:
        return 0
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    print("similarity: " + np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
# Dictionary to store similarities
chunk_similarities = {}
# Calculate cosine similarity for each chunk
for (page, chunk_number), phrases in phrase_embeddings.items():
    similarities = []
    for phrase, embedding in phrases:
        print("phrase: " + phrase)
        phrase_similarities = [cosine_similarity(embedding, query_embedding) for query_embedding in query_embeddings]
        similarities.append(max(phrase_similarities))
    # Choose the highest similarity for each phrase
    average_similarity = np.mean(similarities)
    # Average similarity for the chunk
    chunk_similarities[(page, chunk_number)] = average_similarity
# Get top 5 chunks by similarity
top_chunks = sorted(chunk_similarities.items(), key=lambda x: x[1], reverse=True)[:1]
# Output top 5 chunks
print("Top 5 most relatable chunks:")
selected_chunks = []
for (page, chunk_number), similarity in top_chunks:
    print(f"Page: {page}, Chunk: {chunk_number}, Similarity: {similarity}")
    print(f"Chunk text:\n{page_chunks[page][chunk_number-1]}\n")
    selected_chunks.append(page_chunks[page][chunk_number-1])

context = "\n\n".join(selected_chunks)
prompt = f"Answer the following query based on the provided text:\n\n{context}\n\nQuery: {query}\nAnswer:"
print("prompt: " + prompt)
# Use the Ollama API to get a response
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300
        }
    }
)
response.raise_for_status()
# Extract the answer from the response
answer = response.json()['message']['content'].strip()
# Output the answer
print(f"Answer:\n{answer}")