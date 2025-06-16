
# 📊 Evaluation Notes: RAG-Based Quote Retrieval System

---

## ✅ Evaluation Framework: RAGAS

**RAGAS (Retrieval-Augmented Generation Assessment Suite)** was used to evaluate the system across the following dimensions:

1. **Faithfulness** — Are the generated answers grounded in the retrieved context?
2. **Answer Relevancy** — How relevant is the generated answer to the original query?
3. **Context Precision** — Of the retrieved context, how much is relevant?
4. **Context Recall** — Were all the relevant context elements retrieved?

---

## 📋 Evaluation Setup

- **Query**: `quotes about resilience by women authors`
- **Retrieved Contexts**: Top 10 quotes
- **Generator**: OpenAI `gpt-3.5-turbo`
- **Ground Truth**: `Quotes about resilience from women authors emphasizing strength through adversity.`

---

## 📈 Evaluation Results

| Metric              | Score    | Notes |
|---------------------|----------|-------|
| **Faithfulness**     | 1.000    | Generated output is fully grounded in the quote context |
| **Answer Relevancy** | 0.965    | Very close alignment with query intent |
| **Context Precision**| 0.250    | Many retrieved quotes were loosely relevant or tangential |
| **Context Recall**   | 1.000    | All key relevant quotes were included in the context |

---

## 💡 Observations

- 🔹 The **generator performed very well**, leveraging context to produce fluent, grounded responses.
- 🔹 **High context recall** confirms good coverage of relevant information.
- 🔹 **Low precision** suggests the retriever includes extraneous or weakly related quotes.

---

## 🛠 Recommendations for Improvement

- Tune retrieval threshold or use metadata filters (e.g., author gender, tags) to improve **precision**.
- Consider reducing `top_k` from 10 to 5 to avoid noisy context.
- Integrate a reranker or apply semantic filtering post-retrieval.

---

## ✅ Final Verdict

The RAG pipeline is robust and answers are trustworthy, but **retriever precision** should be optimized to further enhance system performance and generation focus.

