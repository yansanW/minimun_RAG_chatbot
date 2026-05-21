"""
RAGAS Evaluation Script — Google Gemini RAG Pipeline
=====================================================
Run this ONCE after building your RAG app to get quality scores.

Install:
    pip install ragas langchain-google-genai langchain-community chromadb pypdf

Usage:
    python eval_ragas.py

What it measures:
    - faithfulness      : is the answer grounded in the retrieved chunks? (no hallucination)
    - answer_relevancy  : does the answer actually address the question?
    - context_recall    : did retrieval find the chunks needed to answer? (needs ground truth)
    - context_precision : are the retrieved chunks relevant? (no noise)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-key-here") # your-key-here is a placeholder, set your actual key in the environment variable GOOGLE_API_KEY
PDF_PATH = "your_document.pdf"   # your_document.pdf ← point this at your actual PDF

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Test set: questions + ground truth answers ────────────────────────────────
# Write 5–10 Q&A pairs based on your actual document.
# ground_truth = the ideal answer you'd expect. Used for context_recall.
# Be specific — vague ground truths give noisy scores.
test_set = [
    {
        "question": "What is the main strength of this person?",
        "ground_truth": "The paper investigates ..."   # ← fill in from your doc
    },
    {
        "question": "What dataset was used in the experiments?",
        "ground_truth": "The experiments used ..."
    },
    {
        "question": "What were the key findings?",
        "ground_truth": "The key findings were ..."
    },
    # Add 5–7 more pairs for reliable scores
]

# ── Build the same RAG pipeline as your app ───────────────────────────────────
print("Loading and indexing document...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

system_prompt = (
    "Use the following retrieved context to answer the question. "
    "If you don't know, say so.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ── Run each question through the pipeline ────────────────────────────────────
print(f"Running {len(test_set)} questions through pipeline...")

questions, ground_truths, answers, contexts = [], [], [], []

for item in test_set:
    result = qa_chain.invoke({"input": item["question"]})

    questions.append(item["question"])
    ground_truths.append(item["ground_truth"])
    answers.append(result["answer"])
    # RAGAS needs the raw text of each retrieved chunk, not the Document object
    contexts.append([doc.page_content for doc in result["context"]])

    print(f"Q: {item['question'][:60]}...")
    print(f"A: {result['answer'][:120]}...\n")

# ── Wrap models for RAGAS ─────────────────────────────────────────────────────
# RAGAS needs to call the LLM and embeddings internally for its metrics
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ── Build RAGAS dataset ───────────────────────────────────────────────────────
eval_dataset = Dataset.from_dict({
    "question":   questions,
    "answer":     answers,
    "contexts":   contexts,
    "ground_truth": ground_truths,
})

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running RAGAS evaluation...")
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,        # 0–1: is answer grounded in chunks?
        answer_relevancy,    # 0–1: does answer address the question?
        context_recall,      # 0–1: did we retrieve the right chunks?
        context_precision,   # 0–1: are retrieved chunks noise-free?
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)

df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy",
          "context_recall", "context_precision"]].to_string())

print("\nMean scores:")
for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
    score = df[metric].mean()
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {metric:<22} {bar} {score:.3f}")

print("\n" + "="*50)
print("HOW TO IMPROVE LOW SCORES:")
print("  faithfulness low      → chunks don't support the answer → LLM is hallucinating")
print("  answer_relevancy low  → answer is off-topic → improve system prompt")
print("  context_recall low    → retriever missed key chunks → try larger k or reranker")
print("  context_precision low → too much noise retrieved → reduce k or add reranker")
print("="*50)

# ── Save results ──────────────────────────────────────────────────────────────
df.to_csv("ragas_results.csv", index=False)
print("\nSaved detailed results to ragas_results.csv")
