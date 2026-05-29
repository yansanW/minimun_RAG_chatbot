import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


# New imports for the modern chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
import cohere

import tempfile, os

from google.genai import types
from google import genai

# Solved crashes with a 500 INTERNAL error on the second question 
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        # Forces LangChain's string-like TextAccessor object 
        # back into a native Python string to prevent the 500 API crash.
        return super().embed_query(str(text))


st.title("Chat with your PDF")

debug = st.sidebar.checkbox("Debug Mode", value=False)
api_key = st.sidebar.text_input("Google API Key", type="password")
COHERE_API_KEY = st.sidebar.text_input("Cohere API Key (for reranking)", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# ── NEW: initialise chat history in session state ──────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of HumanMessage / AIMessage
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
 

# ── Build the chain once when a file is uploaded ──────────────────────────────
if uploaded_file and api_key :
    
    # Check if we need to build the chain (new file OR chain doesn't exist yet)
    if st.session_state.get("last_file") != uploaded_file.name or st.session_state.qa_chain is None:
        st.session_state.chat_history = []          # reset history for new doc
        st.session_state.last_file = uploaded_file.name
 
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["COHERE_API_KEY"] = COHERE_API_KEY
 
        # Save + load PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Static Chunking (not recommended for best results, but shows the old way)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # Dynamic & Semantic Chunking Strategies
        splitter = SemanticChunker(
            embeddings=SafeGoogleEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,  # CORRECT: Controls chunk size when using "percentile" type
        )

        # Embed + store
        if debug:
            # 2. Instantiate the client using your user's dynamic sidebar API key
            client = genai.Client(api_key=api_key)
            
            # 3. Iterate over the models using the client instance
            for model in client.models.list():
                # 4. Check 'supported_actions' instead of 'supported_generation_methods'
                if 'embedContent' in model.supported_actions:
                    print("Model found:", model.name)
                
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        embeddings = SafeGoogleEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

        texts = [chunk.page_content for chunk in chunks]
        try:
            embeddings_list = embeddings.embed_documents(texts)
        except Exception as e:
            print("Batch embedding failed, trying per-chunk embedding. Error:", e)
            embeddings_list = [embeddings.embed_query(text) for text in texts]

        if debug:
            print(f"Number of embeddings: {len(embeddings_list)}")
            print(f"Number of chunks: {len(chunks)}")

            if len(embeddings_list) != len(chunks):
                print("Warning: Number of embeddings does not match number of chunks!")
        
        # passed embeddings HERE — vectorstore remembers it
        vectorstore = Chroma.from_documents(chunks, embeddings)

        # 1. Base retriever fetches more candidates than before (e.g., k=10)
        # We fetch more because the reranker will narrow it down to the best 3.
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 2. Define your reranker (keeps best 3)
        reranker = CohereRerank(top_n=3, model="rerank-v3.5", cohere_api_key=COHERE_API_KEY)

        # 3. Combine them into a compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )

        # --- MODERN QA CHAIN START ---
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        # ── NEW PROMPT 1: query rewriter ───────────────────────────────────────
        # If the user says "what about the second point?" this rewrites it into
        # a self-contained search query using the chat history as context.
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the chat history and the latest user question, "
             "rewrite the question to be fully self-contained — "
             "so it can be understood without the chat history. "
             "Do NOT answer the question, just rewrite it. "
             "If it's already self-contained, return it unchanged."),
            MessagesPlaceholder("chat_history"),   # ← injects history here
            ("human", "{input}"),
        ])

        # This retriever rewrites the query first, then searches
        # 4. Pass the COMPRESSION retriever / retriever only to the history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, compression_retriever, contextualize_prompt
        )


        # ── NEW PROMPT 2: answer prompt with history ───────────────────────────
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Use the retrieved context below to answer the question. "
             "If you don't know, say so — don't make things up.\n\n"
             "{context}"),
            MessagesPlaceholder("chat_history"),   # ← model sees prior turns
            ("human", "{input}"),
        ])


        combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
 
        # Full chain: history-aware retrieval + answer generation
        st.session_state.qa_chain = create_retrieval_chain(
            history_aware_retriever, combine_docs_chain
        )
 
        st.success(f"Loaded {len(chunks)} chunks. Ask away!")
 
    
    # --- MODERN QA CHAIN END ---


# ── Chat interface ─────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
 
    # Render existing chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)
 
    question = st.chat_input("Ask anything about your PDF...")
 
    if question:
        print("QUESTION:", question)
        print("HISTORY:", st.session_state.chat_history)
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)
 
        # Run the chain — pass full history so it can rewrite the query
        result = st.session_state.qa_chain.invoke({
            "input": question,
            "chat_history": st.session_state.chat_history,  # ← key line
        })
 
        answer = result["answer"]
 
        # Show assistant response
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Source chunks"):
                for doc in result["context"]:
                    st.write(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")
 
        # ── Append this turn to history for next question ──────────────────────
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
 
    # Optional: clear history button
    if st.sidebar.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()