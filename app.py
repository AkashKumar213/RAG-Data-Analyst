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
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Globals
mcp = FastMCP("RAG_Data_Analyst")
os.makedirs("analysis_plots", exist_ok=True)
FAISS_INDEX_DIR = "faiss_index"

# Embeddings & LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3")

# Text splitter for better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def get_db():
    """Load FAISS index or return None."""
    try:
        if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
            return FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, 
                allow_dangerous_deserialization=True
            )
    except Exception as e:
        print(f"Error loading DB: {e}")
    return None

def save_db(db):
    """Save FAISS index."""
    if db:
        try:
            db.save_local(FAISS_INDEX_DIR)
        except Exception as e:
            print(f"Error saving DB: {e}")

def extract_text(file_path: str) -> str:
    """Extract text from PDF or image."""
    if file_path.lower().endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except Exception as e:
            return f"PDF extraction error: {str(e)}"
    
    try:
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        return f"Image extraction error: {str(e)}"

@mcp.tool()
def add_document_to_knowledge_base(file_path: str) -> str:
    """Add document to FAISS knowledge base with proper chunking."""
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    text = extract_text(file_path)
    
    if not text.strip() or "error" in text.lower():
        return f"❌ Could not extract text from: {file_path}"
    
    # Chunk the text
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return f"❌ No valid content in: {file_path}"
    
    # Create metadata
    metadatas = [
        {
            "source": os.path.basename(file_path),
            "chunk": i,
            "total_chunks": len(chunks)
        } 
        for i in range(len(chunks))
    ]
    
    # Load or create FAISS index
    db = get_db()
    if db is None:
        db = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    else:
        db.add_texts(chunks, metadatas=metadatas)
    
    save_db(db)
    return f"✅ Successfully indexed {file_path}\n📊 **{len(chunks)} chunks created**"


@mcp.tool()
def ask_domain_knowledge(query: str, tabular_path=None) -> str:
    """
    RAG query against FAISS knowledge base ONLY.
    Does NOT access any data files or files outside FAISS.
    """
    db = get_db()
    
    if db is None:
        return "📌 Knowledge base is empty. Please upload documents first (PDFs, TXT, or images)."
    
    # Retrieve relevant chunks from FAISS
    docs = db.similarity_search(query, k=5)
    
    if not docs:
        return "❌ No relevant information found in knowledge base for your query."
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        chunk_info = f"Chunk {doc.metadata.get('chunk', '?')}/{doc.metadata.get('total_chunks', '?')}"
        context_parts.append(f"**[{source} - {chunk_info}]**\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer using LLM
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided document context.
If the answer is not in the context, say so clearly.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""
    
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


@mcp.tool()
def get_raw_related_documents(query: str, num_docs: int = 3) -> str:
    """
    Get raw documents from FAISS without generating an answer.
    Used for debugging/verification - shows what's in the knowledge base.
    """
    db = get_db()
    
    if db is None:
        return "empty"  # Signal that DB is empty
    
    # Special case: check if DB has any content
    if query.lower() == "check_db":
        return "Database has documents"
    
    docs = db.similarity_search(query, k=num_docs)
    
    if not docs:
        return "No matching documents found."
    
    result_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        chunk_num = doc.metadata.get('chunk', '?')
        result_parts.append(f"**Document {i}** ({source} - Chunk {chunk_num}):\n{doc.page_content[:300]}...")
    
    return "\n\n---\n\n".join(result_parts)


@mcp.tool()
def analyze_tabular_data(file_path: str) -> str:
    """Analyze and summarize tabular data."""
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        summary = f"📊 **{os.path.basename(file_path)}**\n"
        summary += f"**Shape**: {len(df)} rows × {len(df.columns)} columns\n\n"
        summary += "**Columns**:\n" + "\n".join(f"• {col}: {dtype}" for col, dtype in df.dtypes.items())
        summary += f"\n\n**Statistics**:\n```\n{df.describe().to_string()}\n```"
        
        return summary
    except Exception as e:
        return f"❌ Error analyzing data: {str(e)}"


@mcp.tool()
def create_simple_plot(file_path: str, plot_type: str, x_column: str, y_column: str, title: str = "") -> str:
    """Create a plot from tabular data."""
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # Validate columns
        if x_column not in df.columns or y_column not in df.columns:
            return f"❌ Columns not found. Available: {', '.join(df.columns.tolist())}"
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        if plot_type.lower() == 'line':
            sns.lineplot(data=df, x=x_column, y=y_column)
        elif plot_type.lower() == 'bar':
            sns.barplot(data=df, x=x_column, y=y_column)
        elif plot_type.lower() == 'scatter':
            sns.scatterplot(data=df, x=x_column, y=y_column)
        elif plot_type.lower() == 'histogram':
            plt.hist(df[y_column], bins=30)
        elif plot_type.lower() == 'box':
            sns.boxplot(data=df, y=y_column)
        
        plt.title(title or f"{y_column} vs {x_column}")
        plt.tight_layout()
        
        output_path = f"analysis_plots/plot_{plot_type}_{x_column}_{y_column}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return f"✅ Plot saved: {output_path}"
    except Exception as e:
        return f"❌ Plotting error: {str(e)}"


if __name__ == "__main__":
    mcp.run()