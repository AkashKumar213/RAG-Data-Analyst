import streamlit as st
import os
import tempfile
import pandas as pd
import re  # ← MISSING! Add this
from app import (
    add_document_to_knowledge_base, 
    ask_domain_knowledge, 
    analyze_tabular_data,
    create_simple_plot,
    get_raw_related_documents
)

st.set_page_config(page_title="RAG Data Analyst Agent", page_icon="🤖", layout="wide")

st.title("🤖 RAG Data Analyst Agent (MCP Server Backend)")
st.markdown("This UI interacts with the same core tools exposed by our MCP Server `app.py`.")

# Sidebar
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF/TXT/Image", type=["pdf", "txt", "png", "jpg"])
    
    if st.button("Add to Knowledge Base") and uploaded_file is not None:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            result = add_document_to_knowledge_base(tmp_path)
            st.success(result)
            os.remove(tmp_path)

    st.divider()
    
    st.header("📊 Data Upload")
    data_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="data")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tabular_path" not in st.session_state:
    st.session_state.tabular_path = None
if "df_info" not in st.session_state:
    st.session_state.df_info = ""

# Handle data upload
if data_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{data_file.name}") as tmp:
            tmp.write(data_file.getvalue())
            data_path = tmp.name
        
        if data_file.name.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
            
        st.session_state.tabular_path = data_path
        st.session_state.df_info = f"📊 **Data ready**: {data_file.name} ({len(df)} rows)"
        st.success(st.session_state.df_info)
        
    except Exception as e:
        st.error(f"Data error: {e}")

# Main Chat
st.header("💬 Data Analyst Chat")
st.info("👉 Upload data → Ask: 'line plot sales vs date' → Plot appears here!")

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show embedded plots
        img_matches = re.findall(r'!\[.*?\]\((analysis_plots/.*?)\)', msg["content"])
        for img_path in img_matches:
            if os.path.exists(img_path):
                st.image(img_path, caption="Generated Plot", use_column_width=True)

# Chat input
if prompt := st.chat_input("Ask anything about your data!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing..."):
            response = ask_domain_knowledge(prompt, st.session_state.tabular_path)
            st.markdown(response)
            
            # Show any generated plots
            img_matches = re.findall(r'!\[.*?\]\((analysis_plots/.*?)\)', response)
            for img_path in img_matches:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})