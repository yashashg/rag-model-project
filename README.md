
# 🧠 RAG-Based Semantic Quote Retrieval and Structured QA

> Retrieval-Augmented Generation for semantic search & answer generation using fine-tuned models on quotes.

---

## 📌 Project Overview

This project implements a **semantic quote retrieval system** using **Retrieval-Augmented Generation (RAG)**. The system is built on top of a fine-tuned SentenceTransformer model trained on a curated dataset of quotes, and uses a RAG pipeline (retriever + OpenAI generator) to respond to natural language queries with relevant quotes and structured answers.

> ✅ Fine-tuned model  
> ✅ FAISS vector index  
> ✅ RAG pipeline  
> ✅ RAGAS evaluation  
> ✅ Interactive Streamlit app

---

## 🎯 Objective

To build a system that:
- Retrieves **semantically relevant quotes** based on natural queries
- Leverages **generative models** to create a concise, informative answer
- Evaluates the quality of retrieval + generation using **RAGAS**
- Deploys a **Streamlit UI** for real-time interaction

---

## 📂 Project Structure

```bash
rag-quote-search/
├── cleaned_quotes.csv              # Preprocessed dataset
├── fine_tuned_quote_model/        # Saved SentenceTransformer model
├── rag_pipeline_quotes.ipynb      # Notebook for retrieval, RAG, evaluation
├── rag_pipeline_quotes.ipynb      # Notebook for training, indexing, 
├── app.py                         # Streamlit app for user interaction
├── README.md                      # Project overview and instructions
├── evaluation_notes.md            # Evaluation metrics & analysis
```

---

## 🚀 How to Run the Project

### 🧱 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit faiss-cpu openai pandas sentence-transformers ragas datasets evaluate
```

> ✅ Set your OpenAI key:
```bash
export OPENAI_API_KEY="your-key-here"  # or use .env file
```

---

### 📘 2. Run Jupyter Notebook
```bash
jupyter notebook model_training.ipynb
jupyter notebook rag_pipeline_quotes.ipynb
```
This file covers:
- Loading & cleaning the dataset
- Fine-tuning the model with contrastive loss
- Building FAISS index
- Implementing RAG with OpenAI GPT
- Evaluating with RAGAS

---

### 🌐 3. Launch the Streamlit App
```bash
streamlit run app.py
```
App features:
- Accepts free-form query (e.g., "quotes about courage by women authors")
- Retrieves most relevant quotes
- Generates a coherent response using GPT
- Displays structured output (quote, author, tags, score)

---

## 🧠 Model Architecture

- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Fine-Tuning Strategy**: `MultipleNegativesRankingLoss`
- **Training Data**: Synthetic query–quote pairs
- **Indexing**: FAISS (L2 distance, 384D vectors)
- **Generator**: `gpt-4o-mini` via OpenAI API

---

## 📊 Evaluation (via RAGAS)

| Metric              | Score    |
|---------------------|----------|
| **Faithfulness**     | `1.000`  |
| **Answer Relevancy** | `0.965`  |
| **Context Precision**| `0.250`  |
| **Context Recall**   | `1.000`  |

📌 Interpretation:
- High **faithfulness** → answers are grounded in retrieved quotes  
- High **recall** but lower **precision** → too many quotes retrieved, not all useful  
- Suggests tuning top_k retrieval or adding metadata filters

---

## 🧪 Sample Query

**Input**:
```
quotes about resilience by women authors
```

**Output**:
```json
{
  "quotes": [
    {
      "quote": "You may encounter many defeats, but you must not be defeated.",
      "author": "Maya Angelou",
      "tags": "resilience, women, inspiration",
      "score": 0.83
    },
    ...
  ],
  "generated_answer": "These quotes by inspiring women like Maya Angelou emphasize how resilience is forged through adversity. They reflect determination and inner strength."
}
```

---

## ✅ Design Decisions

- **Contrastive training** for domain-specific semantic retrieval
- **Structured context format** sent to LLM ensures grounded, interpretable generation
- **RAGAS** used for robust, explainable metric-based evaluation
- **Streamlit UI** makes the solution deployable and user-friendly

---

## ⚠️ Challenges & Learnings

- Synthetic data quality strongly influences retrieval performance  
- Retrieval metrics (precision vs recall) can mislead — human inspection still key  
- Prompt engineering matters — context formatting helps guide LLM answers  
- RAG systems require careful calibration between **retriever precision** and **generator fluency**

---

## 🛠 Future Improvements

- Use LLaMA or Mixtral locally (for cost-free generation)
- Advanced query parsing to apply filters (e.g., “only feminist authors”)
- Cache or pre-generate embeddings to boost speed
- Use `Quotient` or `Arize Phoenix` for deeper evaluations

---

## ✍️ Authors & Credits

Built by Yashash Gupta as part of a semantic RAG system assignment using Hugging Face, FAISS, OpenAI, and Streamlit.
