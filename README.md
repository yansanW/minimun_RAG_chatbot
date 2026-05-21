# RAG Chatbot
![alt text](image.png)


# What is RAG?
The full name is **Retrieval-Augmented Generation**. 
Which sounds impressive but is doing a simple job: it stops an LLM from making things up by forcing it to read your documents before answering.

Here’s the problem it solves. 

You build a chatbot using ChatGPT or Claude. A user asks “what’s your return policy?” The LLM doesn’t know your return policy — it was trained on the internet, not your business docs. So it either says “I don’t know” or worse, it confidently makes something up (this is called hallucination, and it’s a real problem).
RAG fixes this by saying: before the LLM answers, go find the relevant part of the document first, then answer based on that.

# Here’s how RAG actually works under the hood. There are two phases:

## Phase 1: Indexing (done once, offline)
You take your documents — a PDF, a website, a FAQ file — and you process them:
1. Split them into smaller chunks (500 words or so)
2. Embed each chunk — convert it into a vector, a list of numbers that captures the meaning
3. Store those vectors in a vector database (ChromaDB, Pinecone, etc.)

## Phase 2: Querying (every time a user asks something)
1. Take the user’s question and embed it — same model, same vector space
2. Search the vector database for chunks whose vectors are closest to the question vector
3. Take the top matching chunks and pass them to the LLM as context
4. The LLM reads those chunks and generates an answer grounded in your actual documents


# Structure Overview
---

```

[ Document ] ──> [ Text Chunks ] ──> [ Embedding Model ] ──> [ Vector Base ]
                                                                      │
                                            ┌─────────────────────────┴───────────────────┐
                                            ▼ (Path ①: Small Data)                        ▼ (Path ②: Large Data)
[ Query ] ──> [ Same Embedding Model ] ──> ① Dot Product                                [ ANN Search (pre-group) ]
                                                │                                              │
                                                ▼                                              ▼
                                         [ Exact Search ]                               ② Dot Product (on neighborhood)
                                                │                                              │
                                                ▼                                              ▼
                                         [ Return Top-K ]                                [ Reranker ]
                                                │                                              │
                                                ▼                                              ▼
                                         [ Return Top-K Chunks ]                        [ Return Top-K Chunks ]
                                                │                                              │
                                                └───────────────────────┬──────────────────────┘
                                                                        │ (Passes selected chunks to LLM)
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │     The LLM      │
                                                              │ (Reads & Generates)│
                                                              └──────────────────┘
                                                                        │
                                                                        ▼
                                                               [ Final Response ]
                                                           (① Exact  OR  ② Routed)
                                                                        │
                                                                        ▼
                                                                 [Evaluation]

```
---
## src/
Core source code.

## 1. Small Data with google gemini api key - simple
```streamlit run src/app_google.py ```

### Overview:
user question → embed → vector search → LLM → answer

Futher level up can be done on:
- Semantic chunking (vs default splitting)
- Conversational memory
- Query rewriting
- Reranker
- Evaluation

## 2. Architected an advanced RAG system that leveraged conversational query-contextualization and Cohere re-ranking to optimize document retrieval relevance by X% (Evaluation with RAGAS).
```streamlit run src/app_google_adv.py # with Google API```

### Overview:
```
user question   → [query rewriting] → vector search
                → [reranking] → LLM with [memory] → answer
                + [evaluation] to know if it's working
```
Extra features:
- Conversational memory
- Query rewriting
- Reranker
- Evaluation (RAGAS Evaluation Script)

#### RAGAS — what the 4 scores actually tell you:
```
faithfulness      → is the answer hallucinated?
                   low score = LLM is making things up beyond the chunks

answer_relevancy  → does the answer address the question?
                   low score = fix your system prompt

context_recall    → did retrieval find the right chunks?
                   low score = increase k, add reranker, try semantic chunking

context_precision → are retrieved chunks clean / noise-free?
                   low score = decrease k, add reranker
```

---

