# KET-RAG: High-Level Architecture

```mermaid
graph TD
  A[Raw Dataset (Text Chunks)] -->|Preprocessing| B[Tokenization & Embedding]
  B -->|Build KNN Graph| C[KNN Graph Construction]
  C -->|Identify Core Chunks| D[PageRank-Based Core Selection]
  
  D -->|Construct Skeleton| E[Knowledge Graph Skeleton (Gₛ)]
  C -->|Build Text-Keyword Graph| F[Text-Keyword Bipartite Graph (Gₖ)]

  E -->|Entity-based Retrieval| G[Graph-Based Context Retrieval]
  F -->|Keyword-based Retrieval| H[Keyword-Based Context Retrieval]

  G -->|Combine Entity & Keyword Contexts| I[Context Fusion]
  H -->|Combine Entity & Keyword Contexts| I
  
  I -->|Feed Context to LLM| J[LLM-Based Answer Generation]
  J -->|Final Response| K[Output Answer]
  
  %% Styling
  style A fill:#f9f,stroke:#333,stroke-width:2px
  style K fill:#ffcc00,stroke:#333,stroke-width:2px
```

## Explanation of the Architecture

### 1️⃣ Raw Dataset (Text Chunks)
- Starts with **unstructured text** from documents.

### 2️⃣ Preprocessing & Tokenization
- Converts raw text into **tokens** and **embeddings** using a transformer model.

### 3️⃣ KNN Graph Construction
- Constructs a **KNN graph** linking text chunks based on **semantic similarity**.

### 4️⃣ Core Chunk Selection (PageRank)
- Uses **PageRank** to **select important chunks** for **Gₛ** (Knowledge Graph Skeleton).

### 5️⃣ Graph Index Construction
- **Gₛ (Entity Graph)**: Extracts structured **(entity, relation, entity)** triplets.
- **Gₖ (Text-Keyword Graph)**: Links **keywords to text chunks** as a lightweight alternative.

### 6️⃣ Dual Retrieval System
- **Graph-Based Retrieval (Gₛ)**: Searches through **entity relations**.
- **Keyword-Based Retrieval (Gₖ)**: Fetches relevant text chunks quickly.

### 7️⃣ Context Fusion
- Merges information from **both graphs** for **optimized retrieval**.

### 8️⃣ LLM-Based Answer Generation
- The **retrieved context** is fed into an **LLM** for final **answer generation**.

### 9️⃣ Final Response
- Outputs a **high-quality response** using **cost-efficient retrieval**.

---

## 🚀 Why This Works Well
✅ **Balances cost & accuracy** via **dual-indexing**.  
✅ **Faster retrieval** with **Gₖ (Keyword Graph)** reducing LLM reliance.  
✅ **Improves generation quality** with **multi-source retrieval**.  

---

📌 *Designed for cost-effective, high-quality retrieval in Graph-RAG applications!*

