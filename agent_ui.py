import streamlit as st
import os
import tempfile
import pandas as pd
import re
import json
from datetime import datetime
from app import (
    add_document_to_knowledge_base, 
    ask_domain_knowledge, 
    analyze_tabular_data,
    create_simple_plot,
    get_raw_related_documents
)
from langchain_ollama import ChatOllama

# ===== HANDLER FUNCTIONS (MUST BE BEFORE MAIN CODE) =====

def build_chat_context(messages, last_n=4):
    history = messages[-last_n:]
    formatted = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted
def handle_data_analysis(prompt: str, tabular_path: str) -> str:
    """Handle data analysis mode with memory + fixed plotting."""

    chat_context = build_chat_context(st.session_state.messages)

    # Load data
    try:
        df = pd.read_csv(tabular_path) if tabular_path.endswith('.csv') else pd.read_excel(tabular_path)
    except Exception as e:
        return f"❌ Error loading data: {str(e)}"

    columns = df.columns.tolist()
    prompt_lower = prompt.lower()

    # =========================
    # 🔥 FORCE PLOT HANDLING (FIX)
    # =========================
    if any(word in prompt_lower for word in ["plot", "chart", "graph"]):
        mentioned_cols = [col for col in columns if col.lower() in prompt_lower]

        if len(mentioned_cols) >= 2:
            return create_simple_plot(
                tabular_path,
                "line",
                mentioned_cols[0],
                mentioned_cols[1],
                f"{mentioned_cols[1]} vs {mentioned_cols[0]}"
            )
        else:
            return f"⚠️ Couldn't detect columns.\nAvailable: {', '.join(columns)}"

    # =========================
    # STATS
    # =========================
    if any(keyword in prompt_lower for keyword in ['stats', 'summary', 'describe', 'info', 'columns']):
        return analyze_tabular_data(tabular_path)

    # =========================
    # 🔥 LLM WITH MEMORY (FIX)
    # =========================
    data_summary = f"**Dataset**: {len(df)} rows, {len(df.columns)} columns\n"
    data_summary += f"**Columns**: {', '.join(columns)}\n\n"
    data_summary += df.head().to_string()

    llm = ChatOllama(model="llama3")

    prompt_text = f"""You are a data analyst.

CHAT HISTORY:
{chat_context}

DATASET:
{data_summary}

QUESTION: {prompt}

ANSWER:"""

    return llm.invoke(prompt_text).content


def handle_document_search(prompt: str) -> str:
    """Handle document search mode (FAISS only) - with privacy."""
    
    db_info = get_raw_related_documents("check_db", num_docs=1)
    
    if "empty" in db_info.lower():
        return "📌 Knowledge base is empty. Please upload documents first (PDF/TXT/Images)."
    
    chat_context = build_chat_context(st.session_state.messages)

    response = ask_domain_knowledge(f"""
    CHAT HISTORY:
    {chat_context}

    QUESTION:
    {prompt}
    """, tabular_path=None)
    
    # ✅ FIX: Don't disclose what documents contain
    # If LLM says "not in context" or mentions document names, hide that info
    if any(phrase in response for phrase in [
        "The answer is NOT in the provided document context",
        "does not mention",
        "no information about",
        "not found in",
        "appears to be"
    ]):
        return "❌ I couldn't find information about that in the uploaded documents. Try asking about different topics."
    
    return response

def handle_internet_search(prompt: str) -> str:
    """Hybrid mode: Works with or without docs."""

    llm = ChatOllama(model="llama3")
    chat_context = build_chat_context(st.session_state.messages)

    # Check if docs exist
    db_info = get_raw_related_documents("check_db", num_docs=1)

    use_docs = "empty" not in db_info.lower()

    retrieved_context = ""
    
    if use_docs:
        try:
            retrieved_context = get_raw_related_documents(prompt, num_docs=3)
        except:
            retrieved_context = ""

    prompt_text = f"""
You are a smart research assistant.

CHAT HISTORY:
{chat_context}

DOCUMENT CONTEXT (if available):
{retrieved_context}

USER QUESTION:
{prompt}

INSTRUCTIONS:
- If document context exists → use it + your knowledge
- If not → answer normally
- Do NOT mention whether docs were used or not

ANSWER:
"""

    response = llm.invoke(prompt_text).content
    return response


# ===== CHAT PERSISTENCE FUNCTIONS =====

def save_chat(filename: str = None) -> str:
    """Save current chat to a JSON file."""
    if not st.session_state.messages:
        return "❌ No messages to save!"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_data = {
        "mode": st.session_state.current_mode,
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    }
    
    filename = filename or f"chat_{timestamp}.json"
    filepath = f"saved_chats/{filename}"
    
    os.makedirs("saved_chats", exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(chat_data, f, indent=2)
    
    return f"✅ Chat saved as: {filename}"


def load_chat(filepath: str) -> bool:
    """Load a chat from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            chat_data = json.load(f)
        
        st.session_state.messages = chat_data.get("messages", [])
        st.session_state.current_mode = chat_data.get("mode", "DATA")
        
        return True
    except Exception as e:
        st.error(f"Error loading chat: {str(e)}")
        return False


def list_saved_chats() -> list:
    """List all saved chats."""
    if not os.path.exists("saved_chats"):
        return []
    return sorted([f for f in os.listdir("saved_chats") if f.endswith(".json")])


# ===== PAGE CONFIG =====

st.set_page_config(page_title="TriSense AI", page_icon="🤖", layout="wide")

st.title("🤖 TriSense AI")
st.markdown("**3 Independent Modes**: Data Analysis | Document Search | Internet Research")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tabular_path" not in st.session_state:
    st.session_state.tabular_path = None
if "df_info" not in st.session_state:
    st.session_state.df_info = ""
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "web_search_mode" not in st.session_state:
    st.session_state.web_search_mode = False
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None
if "auto_save" not in st.session_state:
    st.session_state.auto_save = False

# ===== SIDEBAR: MODE SELECTION =====
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tabular_path = None
        st.session_state.df_info = ""
        st.session_state.editing_index = None
        st.rerun()
    st.header("⚙️ Mode Selection")
    mode = st.radio(
        "Choose Your Analysis Mode:",
        ["📊 Data Analysis", "📚 Document Search", "🌐 Internet Research"],
        index=2 if st.session_state.current_mode is None else ["DATA", "DOC", "WEB"].index(st.session_state.current_mode)
    )
    
    # Map mode name to mode code
    mode_map = {
        "📊 Data Analysis": "DATA",
        "📚 Document Search": "DOC",
        "🌐 Internet Research": "WEB"
    }
    st.session_state.auto_save = st.checkbox("⚡ Auto Save Chat", value=st.session_state.auto_save)
    
    st.session_state.current_mode = mode_map[mode]
    st.session_state.web_search_mode = (st.session_state.current_mode == "WEB")
    
    st.divider()
    
    # ===== MODE 1: DATA ANALYSIS =====
    if st.session_state.current_mode == "DATA":
        st.header("📊 Data Upload")
        st.markdown("Upload CSV/Excel files for analysis & plotting")
        
        data_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="data_upload")
        
        if data_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{data_file.name}", mode='wb') as tmp:
                    tmp.write(data_file.getvalue())
                    data_path = tmp.name
                
                if data_file.name.endswith('.csv'):
                    df = pd.read_csv(data_path)
                else:
                    df = pd.read_excel(data_path)
                    
                st.session_state.tabular_path = data_path
                st.session_state.df_info = f"📊 **{data_file.name}** ({len(df)} rows, {len(df.columns)} cols)"
                st.success(st.session_state.df_info)
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.write(f"**Columns**: {', '.join(df.columns.tolist())}")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # ===== MODE 2: DOCUMENT SEARCH =====
    elif st.session_state.current_mode == "DOC":
        st.header("📚 Knowledge Base")
        st.markdown("Upload PDFs/TXT/Images for semantic search")
        
        uploaded_doc = st.file_uploader("Upload PDF/TXT/Image", type=["pdf", "txt", "png", "jpg"], key="doc_upload")
        
        if st.button("➕ Add to Knowledge Base") and uploaded_doc is not None:
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_doc.name}") as tmp_file:
                    tmp_file.write(uploaded_doc.getvalue())
                    tmp_path = tmp_file.name
                
                result = add_document_to_knowledge_base(tmp_path)
                st.success(result)
                os.remove(tmp_path)
        
        st.info("💡 Once documents are added, search them in the chat below!")
    
    # ===== MODE 3: INTERNET RESEARCH =====
    elif st.session_state.current_mode == "WEB":
        st.header("🌐 Internet Search")
        st.markdown("Search the web for any information")
        st.info("✅ No knowledge base needed - searches internet directly")
        db_info = get_raw_related_documents("check_db", num_docs=1)
        if "empty" in db_info.lower():
            st.warning("📭 No documents found. Using pure internet knowledge.")
        else:
            st.success("📚 Knowledge base detected → Hybrid mode active")

    # ===== CHAT MANAGEMENT (ALL MODES) =====
    st.divider()
    st.header("💾 Chat Management")
    
    st.header("💬 Chats")

    if st.session_state.auto_save:
        saved_chats = list_saved_chats()

        if saved_chats:
            for chat_file in saved_chats[::-1]:
                chat_name = chat_file.replace(".json", "")

                with st.container():
                    cols = st.columns([8, 1])

                    # 👉 Chat Load Button (full clickable feel)
                    with cols[0]:
                        if st.button(
                            f"🗂️ {chat_name}",
                            key=f"load_{chat_file}",
                            use_container_width=True
                        ):
                            if load_chat(f"saved_chats/{chat_file}"):
                                st.rerun()

                    with cols[1]:
                        if st.button("⋮", key=f"menu_{chat_file}"):
                            st.session_state[f"confirm_delete_{chat_file}"] = True

                    if st.session_state.get(f"confirm_delete_{chat_file}", False):
                        if st.button("Delete?", key=f"confirm_{chat_file}"):
                            os.remove(f"saved_chats/{chat_file}")
                            st.rerun()

                    st.markdown("---")  # clean separator
        else:
            st.info("No chats yet")

    else:
        if st.button("💾 Save Chat"):
            st.info(save_chat())

# ===== MAIN CHAT AREA =====
st.divider()

# Mode indicator with clear instructions
if st.session_state.current_mode == "DATA":
    st.markdown("""
    ### 📊 Data Analysis Mode
    **What you can do:**
    - Analyze your uploaded CSV/Excel data
    - Generate plots: "line plot sales vs date"
    - Get statistics: "show me stats" or "describe the data"
    - Ask questions about your data
    """)

elif st.session_state.current_mode == "DOC":
    st.markdown("""
    ### 📚 Document Search Mode
    **What you can do:**
    - Search through uploaded documents
    - Ask questions about document content
    - Semantic search finds relevant sections
    """)

elif st.session_state.current_mode == "WEB":
    st.markdown("""
   ### 🌐 Smart Research Mode (Hybrid)
    - Works without documents
    - If documents exist → auto uses them
    - Combines LLM + retrieval seamlessly
    **What you can do:**
    - Search the internet for any topic
    - Get real-time information
    - No knowledge base required
    """)
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        col1, col2 = st.columns([10, 1])

        with col1:
            st.markdown(msg["content"])

        with col2:
            if st.button("🗑️", key=f"delete_{idx}"):
                st.session_state.messages.pop(idx)

                # Also remove assistant reply if user msg
                if msg["role"] == "user" and idx < len(st.session_state.messages):
                    st.session_state.messages.pop(idx)

                st.rerun()

        # EDIT button for user
        if msg["role"] == "user":
            if st.button("✏️", key=f"edit_{idx}"):
                st.session_state.editing_index = idx

        # Show plots
        img_matches = re.findall(r'analysis_plots/[\w\-\.]+\.png', msg["content"])
        for img_path in img_matches:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

# ✅ FIX: Edit mode for re-asking questions
if st.session_state.editing_index is not None:
    idx = st.session_state.editing_index
    original_prompt = st.session_state.messages[idx]["content"]
    
    st.warning(f"**Editing message {idx + 1}**")
    edited_prompt = st.text_area("Edit your question:", value=original_prompt, key="edit_textarea")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Edit"):
            # Update message
            st.session_state.messages[idx]["content"] = edited_prompt

            # Remove old assistant response
            if idx + 1 < len(st.session_state.messages):
                st.session_state.messages.pop(idx + 1)

            # 🔥 Set trigger instead of running LLM here
            st.session_state.edit_trigger = True
            st.session_state.edit_prompt = edited_prompt
            st.session_state.edit_insert_index = idx + 1

            st.session_state.editing_index = None

            st.rerun()
    
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.editing_index = None
            st.rerun()

# Chat input
if prompt := st.chat_input(f"Ask anything ({['📊 Data', '📚 Docs', '🌐 Web'][['DATA', 'DOC', 'WEB'].index(st.session_state.current_mode)]}):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Processing..."):
            response = ""
            
            # ===== MODE 1: DATA ANALYSIS =====
            if st.session_state.current_mode == "DATA":
                if not st.session_state.tabular_path or not os.path.exists(st.session_state.tabular_path):
                    response = "⚠️ No data file uploaded. Please upload a CSV/Excel file first."
                else:
                    response = handle_data_analysis(prompt, st.session_state.tabular_path)
            
            # ===== MODE 2: DOCUMENT SEARCH (FAISS) =====
            elif st.session_state.current_mode == "DOC":
                response = handle_document_search(prompt)
            
            # ===== MODE 3: INTERNET RESEARCH =====
            elif st.session_state.current_mode == "WEB":
                response = handle_internet_search(prompt)
            
            st.markdown(response)
            
            # Show any generated plots
            img_matches = re.findall(r'analysis_plots/[\w\-\.]+\.png', response)
            for img_path in img_matches:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    if st.session_state.auto_save:
        save_chat()