import os
import glob
import pickle
import pandas as pd
import faiss
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# === Paths & Setup ===
FOLDERS = {
    "ciq": "ciq_files",
    "template": "templates",
    "log": "logs",
    "master_template": "master_templates"
}
FAISS_INDEX_DIR = "faiss_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
router_llm = Ollama(model="llama3")
llm = Ollama(model="llama3.1:8b")
memory = ConversationBufferMemory(return_messages=True)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Document Loaders ===
def load_generic_text_documents(folder, doc_type):
    documents = []
    for filepath in glob.glob(f"{folder}/*"):
        try:
            with open(filepath, "r") as f:
                content = f.read()
            documents.append(Document(page_content=f"[{os.path.basename(filepath)}]\n{content}", metadata={"type": doc_type, "path": filepath}))
        except Exception as e:
            print(f"[!] Error reading {doc_type} {filepath}: {e}")
    return documents

def load_ciq_documents(folder):
    documents = []
    for filepath in glob.glob(f"{folder}/*.xlsx"):
        try:
            df = pd.read_excel(filepath, sheet_name=None)
            content = "\n".join(f"Sheet: {sheet}\n{df[sheet].head(5).to_csv(index=False)}" for sheet in df)
            documents.append(Document(page_content=f"[{os.path.basename(filepath)}]\n{content}", metadata={"type": "ciq", "path": filepath}))
        except Exception as e:
            print(f"[!] Error reading CIQ {filepath}: {e}")
    return documents

# === FAISS ===
def build_faiss_index(docs: List[Document], doc_type: str):
    texts = [doc.page_content for doc in docs]
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    with open(f"{FAISS_INDEX_DIR}/{doc_type}_docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    faiss.write_index(index, f"{FAISS_INDEX_DIR}/{doc_type}_index.faiss")
    print(f"✅ Built FAISS index for {doc_type} ({len(docs)} docs)")

def load_faiss_index(doc_type):
    index = faiss.read_index(f"{FAISS_INDEX_DIR}/{doc_type}_index.faiss")
    with open(f"{FAISS_INDEX_DIR}/{doc_type}_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

def preprocess_all_faiss():
    build_faiss_index(load_ciq_documents(FOLDERS["ciq"]), "ciq")
    build_faiss_index(load_generic_text_documents(FOLDERS["template"], "template"), "template")
    build_faiss_index(load_generic_text_documents(FOLDERS["log"], "log"), "log")
    build_faiss_index(load_generic_text_documents(FOLDERS["master_template"], "master_template"), "master_template")

# === Prompt Templates ===
ciq_prompt = PromptTemplate.from_template("""
You are a telecom assistant. Use the CIQ data below to answer the query.

Context:
{context}

Query:
{query}
""")

template_prompt = PromptTemplate.from_template("""
You are a config assistant. Use this NE template to answer.

Context:
{context}

Query:
{query}
""")

master_template_prompt = PromptTemplate.from_template("""
Use this master template context to respond.

Context:
{context}

Query:
{query}
""")

log_prompt = PromptTemplate.from_template("""
Use the diagnostic log below to help answer the query.

Log:
{context}

Query:
{query}
""")

general_prompt = PromptTemplate.from_template("""
Here is the chat history:

{context}

Query:
{query}
""")

# === Router ===
def route_db(query):
    prompt = PromptTemplate.from_template("""
Classify into: ciq, template, master_template, log, general.
Query: {query}
Category:
""")
    raw = router_llm.invoke(prompt.format(query=query)).strip().lower()
    return raw if raw in FOLDERS else "general"

# === Retrieval & Response ===
def retrieve_and_respond(query, top_k=1):
    category = route_db(query)
    print(f"🔍 Category: {category}")

    if category == "general":
        conversation_history = "\n".join(
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" for msg in memory.buffer[-2:])
        prompt = general_prompt.format(context=conversation_history, query=query)
        response = llm.invoke(prompt)
        memory.save_context({"input": query}, {"output": response})
        return response

    try:
        index, docs = load_faiss_index(category)
        query_vec = EMBED_MODEL.encode([query])
        D, I = index.search(query_vec, top_k)
        doc = docs[I[0][0]] if I[0].size > 0 else None
        context = doc.page_content if doc else "No relevant context."

        conversation_history = "\n".join(
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" for msg in memory.buffer[-2:])
        context_combined = f"{conversation_history}\n{context}"

        if category == "ciq": prompt = ciq_prompt
        elif category == "template": prompt = template_prompt
        elif category == "master_template": prompt = master_template_prompt
        elif category == "log": prompt = log_prompt
        else: prompt = general_prompt

        prompt_filled = prompt.format(context=context_combined, query=query)
        response = llm.invoke(prompt_filled)
        memory.save_context({"input": query}, {"output": response})
        print(f"✅ File used: {doc.metadata['path'] if doc else 'None'}")
        return response

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return "Error in processing."

# === File Upload Endpoint ===
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"📂 File saved: {filepath}")

    # Standardize using LLM
    df = pd.read_excel(filepath) if filename.endswith('.xlsx') else pd.read_csv(filepath)
    std_prompt = f"Standardize this telecom config:\n{df.head(5).to_csv(index=False)}"
    std_output = llm.invoke(std_prompt)

    # Save output
    std_path = os.path.join("standardized", f"std_{filename}.txt")
    os.makedirs("standardized", exist_ok=True)
    with open(std_path, "w") as f:
        f.write(std_output)
    print("🧠 LLM standardization done")

    # Optional: store in FAISS
    if "PCI" in std_output or "TAC" in std_output:
        doc = Document(page_content=std_output, metadata={"type": "master_template", "path": std_path})
        build_faiss_index([doc], "master_template")
        print("📈 Stored in master_template FAISS")

    return send_file(std_path, as_attachment=True)

# === Query Chat Endpoint ===
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is missing"}), 400
    print(f"💬 Query received: {query}")
    response = retrieve_and_respond(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
