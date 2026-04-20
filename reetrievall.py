import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# =========================
# CONFIG
# =========================
PDF_FOLDER = "./pdfs"   # folder with 45 PDFs
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =========================
# LOAD PDFs
# =========================
documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file
        documents.extend(docs)

print(f"Loaded {len(documents)} pages")

# =========================
# CHUNKING
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# =========================
# EMBEDDINGS
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# BASELINE VECTOR STORE (ALL DATA)
# =========================
baseline_db = FAISS.from_documents(chunks, embedding_model)

# =========================
# SIMPLE CATEGORY TAGGING (for optimized)
# =========================
def categorize(text):
    text = text.lower()
    if "temperature" in text or "thermal" in text:
        return "thermal"
    elif "voltage" in text or "electrical" in text:
        return "electrical"
    else:
        return "general"

for c in chunks:
    c.metadata["category"] = categorize(c.page_content)

# =========================
# GROUP BY CATEGORY
# =========================
category_chunks = {}
for c in chunks:
    cat = c.metadata["category"]
    if cat not in category_chunks:
        category_chunks[cat] = []
    category_chunks[cat].append(c)

# create FAISS per category
category_dbs = {
    cat: FAISS.from_documents(docs, embedding_model)
    for cat, docs in category_chunks.items()
}

# =========================
# LLM (Ollama)
# =========================
llm = Ollama(model="mistral")

# =========================
# BASELINE RAG
# =========================
baseline_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=baseline_db.as_retriever(search_kwargs={"k": 4})
)

# =========================
# OPTIMIZED RAG
# =========================
def classify_query(query):
    # simple semantic classification using embedding similarity
    query = query.lower()
    if "temperature" in query or "heat" in query:
        return "thermal"
    elif "voltage" in query:
        return "electrical"
    else:
        return "general"

def optimized_query(query):
    cat = classify_query(query)

    if cat in category_dbs:
        retriever = category_dbs[cat].as_retriever(search_kwargs={"k": 4})
    else:
        retriever = baseline_db.as_retriever(search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)

# =========================
# TEST QUESTIONS (YOU MUST CHANGE THESE FROM PDFs)
# =========================
questions = [
    "How is temperature distributed inside the battery?",
    "What factors affect battery thermal performance?",
    "What is the role of voltage in battery systems?"
]

# =========================
# RUN EXPERIMENT
# =========================
for q in questions:
    print("\n========================")
    print(f"QUESTION: {q}")

    # BASELINE
    start = time.time()
    baseline_answer = baseline_qa.run(q)
    baseline_time = time.time() - start

    print("\n--- Baseline Answer ---")
    print(baseline_answer)
    print(f"Time: {baseline_time:.3f}s")

    # OPTIMIZED
    start = time.time()
    optimized_answer = optimized_query(q)
    optimized_time = time.time() - start

    print("\n--- Optimized Answer ---")
    print(optimized_answer)
    print(f"Time: {optimized_time:.3f}s")