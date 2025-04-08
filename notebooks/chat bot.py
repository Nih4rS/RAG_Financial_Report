import os
import json
import subprocess
import sys
from time import sleep

# --- Auto-install required packages ---
def install_if_needed(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg_name])

# Basic packages for our pipeline
packages = [
    ("langchain", None),
    ("langchain-community", None),
    ("langchain-huggingface", "langchain_huggingface"),
    ("transformers", None),
    ("sentence_transformers", "sentence_transformers"),
    ("faiss-cpu", None)
]
for pkg, imp in packages:
    install_if_needed(pkg, imp)

# --- Import libraries ---
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# Try importing the new HuggingFacePipeline from langchain_huggingface; fallback if needed.
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import transformers

# --- Set up free LLM using HuggingFacePipeline ---
model_name = "google/flan-t5-small"
pipe = transformers.pipeline(
    "text2text-generation",
    model=model_name,
    tokenizer=model_name,
    max_length=256,
    do_sample=True,
    temperature=0.1  # Temperature set to a positive value to avoid errors
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Set up embeddings for vector store and similarity ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Set up cross-encoder for reranking ---
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Function to load SEC 10-K documents from JSON files ---
def load_documents(data_dir="sec_10k_data"):
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        filings = json.load(f)
                        for filing in filings:
                            text = (
                                f"Form: {filing.get('form_type')}. "
                                f"Date: {filing.get('filing_date')}. "
                                f"Accession: {filing.get('accession_no')}. "
                                f"URL: {filing.get('doc_url')}."
                            )
                            metadata = {"source": file_path}
                            documents.append({"text": text, "metadata": metadata})
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents

docs = load_documents()
if not docs:
    print("No documents loaded. Check your 'sec_10k_data' directory.")
else:
    print(f"Loaded {len(docs)} documents.")

# --- Create FAISS Vector Store ---
try:
    texts = [doc["text"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    print("Vector store created successfully.")
except Exception as e:
    print("Error creating vector store:", e)

# --- Set up the RetrievalQA chain ---
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    print("RetrievalQA chain is ready.")
except Exception as e:
    print("Error setting up the QA chain:", e)

# --- Function to compute cosine similarity ---
def compute_similarity(answer, documents):
    try:
        answer_embedding = similarity_model.encode(answer, convert_to_tensor=True)
        sims = []
        for doc in documents:
            doc_embedding = similarity_model.encode(doc, convert_to_tensor=True)
            cosine_sim = util.cos_sim(answer_embedding, doc_embedding)
            sims.append(cosine_sim.item())
        if sims:
            avg_sim = sum(sims) / len(sims)
            return avg_sim, sims
        else:
            return 0, []
    except Exception as e:
        print("Error computing similarity:", e)
        return 0, []

# --- Function for reranking retrieved documents ---
def rerank_documents(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]

# --- Function for query decomposition ---
def decompose_query(query):
    prompt = f"Decompose this query into sub-questions to get more detailed answers: '{query}'"
    subqueries = llm(prompt)
    return [sq.strip() for sq in subqueries.split("\n") if sq.strip()]

# --- Interactive Query Loop with React Reflection ---
def interactive_query():
    print("\n--- RAG Query System with Reranking, Decomposition, and Reflection ---")
    query = input("Enter your query about 10-K filings: ").strip()
    
    # Optionally decompose complex queries
    if len(query.split()) > 10:
        sub_queries = decompose_query(query)
        print("\nSub-queries generated:")
        for idx, sub in enumerate(sub_queries, 1):
            print(f"{idx}. {sub}")
    
    # Retrieve a broader set of documents.
    initial_docs = vector_store.similarity_search(query, k=10)
    if not initial_docs:
        print("No relevant documents found.")
        return
    
    # Rerank documents using the cross-encoder.
    top_docs = rerank_documents(query, initial_docs, top_k=3)
    
    # Generate an answer using the RetrievalQA chain.
    answer = qa_chain.run(query)
    print("\nLLM Answer:\n", answer)
    
    # Display top retrieved documents.
    print("\nTop Retrieved Documents after Reranking:")
    for i, doc in enumerate(top_docs, start=1):
        print(f"\nDocument {i} (Source: {doc.metadata.get('source', 'N/A')}):")
        print(doc.page_content)
    
    # Compute cosine similarity as a proxy for faithfulness.
    retrieved_texts = [doc.page_content for doc in top_docs]
    avg_similarity, sims = compute_similarity(answer, retrieved_texts)
    print("\nFaithfulness Metrics:")
    print("Average Cosine Similarity:", avg_similarity)
    print("Individual Similarities:", sims)

interactive_query()

# Future improvements:
# - Fine-tune the LLM on your SEC filings data.
# - Optimize query formulation and ask follow-up questions automatically if details are missing.
# - Track and display top-searched topics for further analysis.
