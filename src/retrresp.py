import os
import glob
import pickle
import pandas as pd
import faiss
from typing import List, Dict, Optional
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
    "master_template": "master_templates",
    "standard_ciq": "standard_ciq"
}
os.makedirs(FOLDERS["standard_ciq"], exist_ok=True)
FAISS_INDEX_DIR = "faiss_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
router_llm = Ollama(model="llama3")
llm = Ollama(model="llama3.1:8b")

# === Standard CIQ Management ===
STANDARD_CIQ_FILE = os.path.join(FOLDERS["standard_ciq"], "standard_ciq.xlsx")

def load_standard_ciq() -> Optional[pd.DataFrame]:
    if os.path.exists(STANDARD_CIQ_FILE):
        return pd.read_excel(STANDARD_CIQ_FILE)
    return None

def save_standard_ciq(df: pd.DataFrame):
    df.to_excel(STANDARD_CIQ_FILE, index=False)
    # Rebuild FAISS index after updating standard CIQ
    preprocess_all_faiss()

def get_standard_columns() -> List[str]:
    standard_df = load_standard_ciq()
    return list(standard_df.columns) if standard_df is not None else []

# === CIQ Standardization Functions ===
def standardize_ciq(input_file_path: str) -> Dict:
    """
    Standardize the input CIQ file to match the standard format.
    Returns a dict with:
    - 'output_df': Standardized DataFrame
    - 'missing_columns': Columns in input but not in standard
    - 'extra_columns': Columns in standard but not in input
    """
    standard_df = load_standard_ciq()
    if standard_df is None:
        raise ValueError("No standard CIQ template available")
    
    input_df = pd.read_excel(input_file_path)
    
    # Find column mappings (case insensitive)
    input_cols = [col.lower() for col in input_df.columns]
    standard_cols = [col.lower() for col in standard_df.columns]
    
    # Create mapping from input to standard columns
    column_mapping = {}
    missing_in_standard = []
    missing_in_input = []
    
    for std_col in standard_cols:
        if std_col in input_cols:
            column_mapping[std_col] = std_col
        else:
            missing_in_input.append(std_col)
    
    for in_col in input_cols:
        if in_col not in standard_cols:
            missing_in_standard.append(in_col)
    
    # Create standardized output
    output_df = pd.DataFrame(columns=standard_df.columns)
    
    # Map values from input to output
    for std_col in standard_df.columns:
        std_col_lower = std_col.lower()
        if std_col_lower in column_mapping:
            in_col = column_mapping[std_col_lower]
            output_df[std_col] = input_df[in_col]
    
    return {
        "output_df": output_df,
        "missing_columns": missing_in_standard,
        "extra_columns": missing_in_input
    }

def handle_ciq_standardization(query: str, file_path: str) -> str:
    """
    Handle the CIQ standardization process including user confirmation
    for adding new columns to the standard template.
    """
    try:
        standardization_result = standardize_ciq(file_path)
        output_df = standardization_result["output_df"]
        missing_cols = standardization_result["missing_columns"]
        extra_cols = standardization_result["extra_columns"]
        
        response_parts = []
        
        if missing_cols:
            response_parts.append(
                f"⚠️ These columns in your file aren't in our standard: {', '.join(missing_cols)}\n"
                f"Should I add them to our standard template? (yes/no)"
            )
            memory.save_context(
                {"input": query}, 
                {"output": "\n".join(response_parts)}
            )
            return {
                "type": "missing_columns",
                "message": "\n".join(response_parts),
                "missing_columns": missing_cols,
                "file_path": file_path
            }
        
        if extra_cols:
            response_parts.append(
                f"Note: These standard columns weren't in your file: {', '.join(extra_cols)}"
            )
        
        # Save the standardized file
        output_path = os.path.join(FOLDERS["ciq"], "standardized_" + os.path.basename(file_path))
        output_df.to_excel(output_path, index=False)
        
        response_parts.append(
            f"✅ Standardized CIQ created: {output_path}\n"
            f"The file has been standardized to match our template."
        )
        
        return {
            "type": "file",
            "message": "\n".join(response_parts),
            "file_path": output_path
        }
        
    except Exception as e:
        return {
            "type": "error",
            "message": f"❌ Error standardizing CIQ: {str(e)}"
        }

def update_standard_ciq(new_columns: List[str], file_path: str) -> Dict:
    """
    Update the standard CIQ template with new columns and re-standardize the input file.
    """
    try:
        standard_df = load_standard_ciq()
        input_df = pd.read_excel(file_path)
        
        # Add new columns to standard
        for col in new_columns:
            if col.lower() not in [c.lower() for c in standard_df.columns]:
                standard_df[col] = ""  # Add empty column
        
        # Save updated standard
        save_standard_ciq(standard_df)
        
        # Now re-standardize the original file
        standardization_result = standardize_ciq(file_path)
        output_df = standardization_result["output_df"]
        
        # Save the standardized file
        output_path = os.path.join(FOLDERS["ciq"], "standardized_" + os.path.basename(file_path))
        output_df.to_excel(output_path, index=False)
        
        return {
            "type": "file",
            "message": f"✅ Standard template updated and new standardized file created: {output_path}",
            "file_path": output_path
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"❌ Error updating standard CIQ: {str(e)}"
        }

# === Loaders ===
def load_generic_text_documents(folder, doc_type):
    documents = []
    for filepath in glob.glob(f"{folder}/*"):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r") as f:
                content = f.read()
            documents.append(Document(
                page_content=f"[{filename}]\n{content}",
                metadata={"type": doc_type, "path": filepath}
            ))
        except Exception as e:
            print(f"[!] Error reading {doc_type} {filepath}: {e}")
    return documents

def load_ciq_documents(folder):
    documents = []
    for filepath in glob.glob(f"{folder}/*.xlsx"):
        filename = os.path.basename(filepath)
        try:
            df = pd.read_excel(filepath, sheet_name=None)
            content = "\n".join(
                f"Sheet: {sheet}\n{df[sheet].head(5).to_csv(index=False)}"
                for sheet in df
            )
            documents.append(Document(
                page_content=f"[{filename}]\n{content}",
                metadata={"type": "ciq", "path": filepath}
            ))
        except Exception as e:
            print(f"[!] Error reading CIQ {filepath}: {e}")
    return documents

# === FAISS Embedding Indexing ===
def build_faiss_index(docs: List[Document], doc_type: str):
    texts = [doc.page_content for doc in docs]
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    with open(f"{FAISS_INDEX_DIR}/{doc_type}_docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    faiss.write_index(index, f"{FAISS_INDEX_DIR}/{doc_type}_index.faiss")
    print(f"✅ Built FAISS index for {doc_type} ({len(docs)} docs)")

def preprocess_all_faiss():
    build_faiss_index(load_ciq_documents(FOLDERS["ciq"]), "ciq")
    build_faiss_index(load_generic_text_documents(FOLDERS["template"], "template"), "template")
    build_faiss_index(load_generic_text_documents(FOLDERS["log"], "log"), "log")
    build_faiss_index(load_generic_text_documents(FOLDERS["master_template"], "master_template"), "master_template")

def load_faiss_index(doc_type):
    index = faiss.read_index(f"{FAISS_INDEX_DIR}/{doc_type}_index.faiss")
    with open(f"{FAISS_INDEX_DIR}/{doc_type}_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

# === Router ===
def route_db(query):
    prompt = PromptTemplate.from_template("""
    Classify the user query into one of the categories:

    - ciq: Excel CIQ files with columns like PCI, TAC, etc.
    - template: Config templates generated from CIQ.
    - master_template: Global/merged/unified templates.
    - log: Diagnostic log files or error traces.
    - general: Chat or unrelated to files.

    Query: "{query}"
    Only respond with one of: ciq, template, master_template, log, general.
    Category:
    """)
    raw = router_llm.invoke(prompt.format(query=query)).strip().lower()
    return raw if raw in ["ciq", "template", "master_template", "log"] else "general"

# === Prompts per Category ===
ciq_prompt = PromptTemplate.from_template("""
You are a telecom assistant. Use the CIQ (Customer Information Questionnaire) sheet data below to answer the user query.

Context:
{context}

User Query:
{query}

Answer in a structured and helpful way:
""")

template_prompt = PromptTemplate.from_template("""
You are a config assistant. Use the following NE template (generated from a CIQ) to answer the question.

Template Context:
{context}

Query:
{query}

Response:
""")

master_template_prompt = PromptTemplate.from_template("""
You are a deployment assistant. The context below contains a **global or merged NE master template**. Use it to answer the query.

Master Template:
{context}

User Query:
{query}

Answer with high-level clarity:
""")

log_prompt = PromptTemplate.from_template("""
You are a diagnostics assistant. The context contains a system or error log. Use it to troubleshoot or respond.

Log Context:
{context}

User Query:
{query}

Provide an insightful and actionable answer:
""")

general_prompt = PromptTemplate.from_template("""
You are an assistant. The context contains chat history. 

Past Conversation:
{context}

User Query:
{query}

Provide a crisp and short answer:
""")

# === Memory (Chat History) ===
memory = ConversationBufferMemory(return_messages=True)

# === Query Execution ===
def retrieve_and_respond(query: str, file_path: str = None) -> Dict:
    """
    Main function to handle queries and file processing.
    Returns a dict with:
    - type: "text", "file", "missing_columns", or "error"
    - message: Response message
    - file_path: Path to generated file (if applicable)
    - missing_columns: List of missing columns (if applicable)
    """
    if file_path and file_path.lower().endswith('.xlsx'):
        # Handle file upload (CIQ standardization)
        return handle_ciq_standardization(query, file_path)
    
    # Check if this is a follow-up about missing columns
    last_message = memory.buffer[-1].content if memory.buffer else ""
    if "should I add them to our standard template?" in last_message and "yes" in query.lower():
        # Extract missing columns from context
        missing_cols = []
        for msg in memory.buffer:
            if "aren't in our standard" in msg.content:
                parts = msg.content.split(":")[1].split("\n")[0].strip()
                missing_cols = [col.strip() for col in parts.split(",")]
                break
        
        if missing_cols:
            # Get the original file path from memory
            original_file_path = None
            for msg in memory.buffer:
                if "file_path" in msg.additional_kwargs:
                    original_file_path = msg.additional_kwargs["file_path"]
                    break
            
            if original_file_path:
                return update_standard_ciq(missing_cols, original_file_path)
        
        return {
            "type": "error",
            "message": "Couldn't determine which columns to add. Please try again."
        }
    
    # Normal query processing
    category = route_db(query)
    print(f"\n🔍 Routed to: {category}")

    if category == "general":
        conversation_history = ""
        for msg in memory.buffer:
            role = "User" if msg.type == "human" else "Assistant"
            conversation_history += f"{role}: {msg.content}\n"
        
        prompt_with_memory = general_prompt.format(
            context=conversation_history,
            query=query
        )
        
        response = llm.invoke(prompt_with_memory)
        memory.save_context({"input": query}, {"output": response})
        return {
            "type": "text",
            "message": response
        }

    try:
        index, docs = load_faiss_index(category)
        query_vec = EMBED_MODEL.encode([query])
        distances, indices = index.search(query_vec, 1)
        selected_doc = docs[indices[0][0]] if indices[0].size > 0 else None
        context = selected_doc.page_content if selected_doc else "No relevant context found."

        # Add previous conversation from memory
        conversation_history = ""
        for msg in memory.buffer:
            role = "User" if msg.type == "human" else "Assistant"
            conversation_history += f"{role}: {msg.content}\n"

        # Select appropriate prompt
        if category == "ciq":
            prompt = ciq_prompt
        elif category == "template":
            prompt = template_prompt
        elif category == "master_template":
            prompt = master_template_prompt
        elif category == "log":
            prompt = log_prompt
        else:
            prompt = general_prompt

        prompt_with_memory = prompt.format(
            context=f"{conversation_history}\n{context}",
            query=query
        )
        
        response = llm.invoke(prompt_with_memory)
        memory.save_context({"input": query}, {"output": response})
        
        return {
            "type": "text",
            "message": response,
            "source_doc": selected_doc.metadata["path"] if selected_doc else None
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"❌ Retrieval or LLM failed: {str(e)}"
        }
