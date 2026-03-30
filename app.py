import os
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from mcp.server.fastmcp import FastMCP
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Globals
mcp = FastMCP("RAG_Data_Analyst")
os.makedirs("analysis_plots", exist_ok=True)
FAISS_INDEX_DIR = "faiss_index"

# Embeddings & LLM (Ollama only - no OpenAI)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3")

def get_db():
    """Load FAISS index or return None."""
    try:
        if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
            return FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, 
                allow_dangerous_deserialization=True
            )
    except Exception:
        pass
    return None

def save_db(db):
    """Save FAISS index."""
    if db:
        db.save_local(FAISS_INDEX_DIR)

def extract_text(file_path: str) -> str:
    """Extract text from PDF or image."""
    if file_path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text
    try:
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception:
        return "Text extraction failed."

@mcp.tool()
def add_document_to_knowledge_base(file_path: str) -> str:
    """Add document to FAISS knowledge base."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    text = extract_text(file_path)
    if not text.strip():
        return f"No text extracted from: {file_path}"
    
    db = get_db() or FAISS.from_texts(
        [text], embeddings, metadatas=[{"source": os.path.basename(file_path)}]
    )
    db.add_texts([text], metadatas=[{"source": os.path.basename(file_path)}])
    save_db(db)
    return f"✅ Indexed {file_path}"



@mcp.tool()
def ask_domain_knowledge(query, tabular_path=None):
    """RAG + agentic plotting."""
    # RAG
    db = get_db()
    if db:
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\nAnswer:").content
    else:
        response = "Upload documents first."
    
    # Auto-plotting
    if tabular_path and os.path.exists(tabular_path):
        for pattern in [
            r'(line|bar|scatter)\s+(?:chart|plot)\s+(.+?)(?:\s+from\s+data)?',
            r'(line|bar|scatter)\s+(.+)',
        ]:
            match = re.search(pattern, query.lower())
            if match:
                plot_type, cols_text = match.groups()
                cols = [col.strip() for col in cols_text.split(',')]
                
                try:
                    df = pd.read_csv(tabular_path) if tabular_path.endswith('.csv') else pd.read_excel(tabular_path)
                    if len(cols) >= 2 and cols[0] in df.columns and cols[1] in df.columns:
                        x_col, y_col = cols[:2]
                        plot_name = f"auto_plot_{x_col}_{y_col}_{plot_type}.png"
                        create_simple_plot(tabular_path, plot_type, x_col, y_col, f"{y_col} vs {x_col}")
                        return f"{response}\n\n📊 **Plot**: analysis_plots/{plot_name}"
                except Exception:
                    pass
    
    return response

@mcp.tool()
def get_raw_related_documents(query: str, num_docs: int = 3) -> str:
    """Raw FAISS document retrieval."""
    db = get_db()
    if not db:
        return "Knowledge base empty."
    
    docs = db.similarity_search(query, k=num_docs)
    return "\n\n".join(
        f"Doc {i+1} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

@mcp.tool()
def analyze_tabular_data(file_path: str) -> str:
    """Tabular data summary."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    summary = f"📊 {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} cols\n\n"
    summary += "Columns:\n" + "\n".join(f"• {col}: {dtype}" for col, dtype in df.dtypes.items())
    return summary + f"\n\nStats:\n{df.describe().to_string()}"

@mcp.tool()
def create_simple_plot(file_path: str, plot_type: str, x_column: str, y_column: str, title: str = "") -> str:
    """Generate plot."""
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    if plot_type.lower() == 'line':
        sns.lineplot(data=df, x=x_column, y=y_column)
    elif plot_type.lower() == 'bar':
        sns.barplot(data=df, x=x_column, y=y_column)
    elif plot_type.lower() == 'scatter':
        sns.scatterplot(data=df, x=x_column, y=y_column)
    
    plt.title(title or f"{y_column} vs {x_column}")
    plt.tight_layout()
    
    output_path = f"analysis_plots/plot_{plot_type}_{x_column}_{y_column}.png"
    plt.savefig(output_path)
    plt.close()
    
    return f"Plot saved: {output_path}"

if __name__ == "__main__":
    mcp.run()