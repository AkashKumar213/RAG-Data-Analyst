import os
import shutil
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Initialize MCP Server
mcp = FastMCP("RAG_Data_Analyst")

# Setup embeddings and directories
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
os.makedirs("uploaded_docs", exist_ok=True)
os.makedirs("analysis_plots", exist_ok=True)
FAISS_INDEX_DIR = "faiss_index"

# LLM setup
USE_OLLAMA = True
llm = ChatOllama(model="llama3") if USE_OLLAMA else ChatOpenAI(model="gpt-4o")

def get_db():
    try:
        if os.path.exists(FAISS_INDEX_DIR) and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
            return FAISS.load_local(
                FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
    except Exception as e:
        print(f"Index loading issue: {e}")
    return None

def save_db(db):
    if db:
        db.save_local(FAISS_INDEX_DIR)

def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else: # Fallback to image OCR
        try:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"Error extracting text: {str(e)}"

@mcp.tool()
def add_document_to_knowledge_base(file_path: str) -> str:
    """
    Reads a document (PDF or Image) from the given file_path locally, extracts text,
    and adds it to the persistent FAISS knowledge base.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    text = extract_text(file_path)
    if not text.strip():
        return f"No text could be extracted from: {file_path}"
        
    db = get_db()
    
    # Store with metadata for tracking
    file_name = os.path.basename(file_path)
    
    if db is None:
        db = FAISS.from_texts([text], embeddings, metadatas=[{"source": file_name}])
    else:
        db.add_texts([text], metadatas=[{"source": file_name}])
    save_db(db)
    
    return f"Successfully read and indexed {file_path} into the knowledge base."

@mcp.tool()
def ask_domain_knowledge(query, tabular_path=None):
    """Main RAG query handler with auto-plotting."""
    
    # REAL RAG LOGIC (replace the broken placeholder)
    db = get_db()
    if db:
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Using this context from documents, answer the question.

Context: {context}

Question: {query}

Answer:"""
        rag_response = llm.invoke(prompt).content
    else:
        rag_response = "No documents in knowledge base yet. Upload PDFs first."
    
    # PLOT DETECTION
    if tabular_path and os.path.exists(tabular_path):
        plot_patterns = [
            r'(line|bar|scatter)\s+(?:chart|plot)\s+(?:for|of)\s+(.+?)(?:\s+from\s+data)?',
            r'make\s+(line|bar|scatter)\s+(?:plot|chart)\s+(.+)',
            r'show\s+(line|bar|scatter)\s+(.+)',
        ]
        
        for pattern in plot_patterns:
            match = re.search(pattern, query.lower())
            if match:
                plot_type = match.group(1)
                cols_text = match.group(2).strip(', ')
                cols = [col.strip() for col in cols_text.split()]
                
                try:
                    df = pd.read_csv(tabular_path) if tabular_path.endswith('.csv') else pd.read_excel(tabular_path)
                    valid_cols = [c for c in cols if c in df.columns]
                    
                    if len(valid_cols) >= 2:
                        x_col, y_col = valid_cols[0], valid_cols[1]
                        plot_name = f"auto_plot_{x_col}_{y_col}_{plot_type}.png"
                        plot_result = create_simple_plot(tabular_path, plot_type, x_col, y_col, f"{y_col} vs {x_col}")
                        
                        return f"{rag_response}\n\n📊 **Generated {plot_type} plot:**\n![{plot_type} plot](analysis_plots/{plot_name})"
                except Exception as e:
                    rag_response += f"\n\n⚠️ Plot error: {str(e)}"
    
    return rag_response

@mcp.tool()
def get_raw_related_documents(query: str, num_docs: int = 3) -> str:
    """
    Retrieves the raw text and source from the knowledge base related to the query.
    Use this when the user just wants the related doc rather than an LLM synthesis.
    """
    db = get_db()
    if db is None:
        return "The knowledge base is empty. Please read documents first."
        
    docs = db.similarity_search(query, k=num_docs)
    
    result = f"Found {len(docs)} related document chunks:\n\n"
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        result += f"--- Document {i+1} (Source: {source}) ---\n"
        result += doc.page_content + "\n\n"
        
    return result

@mcp.tool()
def analyze_tabular_data(file_path: str) -> str:
    """
    Data Analyst Tool: Reads a CSV or Excel file and returns a summary of the data 
    (size, columns, data types, basic stats) for analytics perspective.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format. Please provide a CSV or Excel file."
            
        summary = f"Data Analysis Summary for {file_path}:\n"
        summary += f"- Total Rows: {len(df)}\n"
        summary += f"- Total Columns: {len(df.columns)}\n\n"
        summary += "Columns and Data Types:\n"
        for col, dtype in df.dtypes.items():
            summary += f"  - {col}: {dtype}\n"
            
        summary += "\nBasic Statistics (Numeric columns):\n"
        summary += df.describe().to_string()
        
        return summary
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

@mcp.tool()
def create_simple_plot(file_path: str, plot_type: str, x_column: str, y_column: str, title: str = "Data Plot") -> str:
    """
    Data Analyst Tool: Creates a simple plot from a CSV or Excel file. 
    plot_type must be either 'line', 'bar', or 'scatter'.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
        
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format."
            
        if x_column not in df.columns or y_column not in df.columns:
            return f"Columns '{x_column}' or '{y_column}' not found in data."
            
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        if plot_type.lower() == 'line':
            sns.lineplot(data=df, x=x_column, y=y_column)
        elif plot_type.lower() == 'bar':
            sns.barplot(data=df, x=x_column, y=y_column)
        elif plot_type.lower() == 'scatter':
            sns.scatterplot(data=df, x=x_column, y=y_column)
        else:
            return "Unsupported plot_type. Use 'line', 'bar', or 'scatter'."
            
        plt.title(title)
        plt.tight_layout()
        
        output_name = f"plot_{plot_type}_{x_column}_{y_column}.png".replace("/", "_").replace(" ", "")
        output_path = os.path.join("analysis_plots", output_name)
        
        plt.savefig(output_path)
        plt.close()
        
        # We assume local execution where the user has file access
        return f"Plot successfully created and saved at: {os.path.abspath(output_path)}"
    except Exception as e:
        return f"Error creating plot: {str(e)}"

if __name__ == "__main__":
    mcp.run()
