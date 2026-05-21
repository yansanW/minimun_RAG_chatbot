import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# New imports for the modern chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile, os

from google.genai import types
import google.generativeai as genai

st.title("Chat with your PDF")

api_key = st.sidebar.text_input("Google API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
debug = st.sidebar.checkbox("Debug Mode", value=False)

if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # Save + load PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Chunk it
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Embed + store
    if debug:
        genai.configure(api_key=api_key)
        for model in genai.list_models():
            if 'embedContent' in model.supported_generation_methods:
                print("Model found:",model.name)
            
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
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

    # --- MODERN QA CHAIN START ---
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    # Define how the LLM should answer
    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Build the chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 3}),  # So when retriever is called, it already knows, which embedding model to use for the query
        combine_docs_chain
    )
    # --- MODERN QA CHAIN END ---

    question = st.text_input("Ask anything about your PDF...")
    if question:
        # Note: Change "query" to "input" for the new chain
        result = qa_chain.invoke({"input": question})
        
        # Note: New keys are "answer" and "context"
        st.write(result["answer"])

        with st.expander("See source chunks"):
            for doc in result["context"]:
                st.write(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")
