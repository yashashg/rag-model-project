import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load model and data
st.set_page_config(page_title="Semantic Quote Search")
st.title("🧠 Semantic Quote Retrieval with RAG")
key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("Please set the OPENAI_API_KEY environment variable to use this app.")
    st.stop()

@st.cache_resource
def load_data():
    df = pd.read_csv("cleaned_quotes.csv")
    model = SentenceTransformer("./model/fine_tuned_quote_model")
    corpus = df["combined"].tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df, model, index, embeddings

df, model, index, embeddings = load_data()

def retrieve_quotes(query, top_k=5, similarity_threshold=0.75):
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        similarity = 1 - (score / 4)  # approximate cosine similarity from L2
        if similarity >= similarity_threshold:
            results.append({
                "quote": df.iloc[idx]["quote"],
                "author": df.iloc[idx]["author"],
                "tags": df.iloc[idx]["tags"],
                "score": round(similarity, 3)
            })
    return results

def generate_answer(query, context_quotes):
    # Initialize client using environment variable
    client = OpenAI(api_key=key)

    # Create a prompt using retrieved quotes
    context_text = "\n".join([f"- {q['quote']} ({q['author']})" for q in context_quotes])
    
    prompt = f"""
You are a helpful assistant. A user asked the following query:
"{query}"

Here are some relevant quotes:
{context_text}

Based on these quotes, provide a helpful and relevant response or summary:
"""

    # Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# --- UI ---
query = st.text_input("Enter a query (e.g., 'quotes about courage by women authors')")

if st.button("Search") and query:
    with st.spinner("Retrieving relevant quotes..."):
        quotes = retrieve_quotes(query)
        if not quotes:
            st.warning("No relevant quotes found.")
        else:
            st.subheader("🔍 Retrieved Quotes:")
            for q in quotes:
                st.markdown(f"> *{q['quote']}*  \n — **{q['author']}** | Tags: `{q['tags']}` | Similarity: `{q['score']}`")

            st.subheader("💬 Generated Answer:")
            answer = generate_answer(query, quotes)
            st.success(answer)

            st.subheader("📦 Structured JSON Output:")
            st.json({
                "query": query,
                "quotes": quotes,
                "generated_answer": answer
            })
