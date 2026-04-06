# 🤖 RAG Data Analyst Agent

**Production-ready Streamlit app** that transforms natural language into data analysis + charts. Ask "line plot sales vs date" → instant visualization!

## 🚀 Features
- **3 Independent Modes** - Data Analysis | Document Search | Internet Research
- **FAISS RAG** - Semantic search across PDFs/Images (Document Search mode)
- **Agentic Plotting** - "scatter plot revenue vs models" → auto-generates charts (Data Analysis mode)
- **Excel/CSV Analysis** - Upload data → conversational analytics (Data Analysis mode)
- **Ollama LLM** - Local Llama3 inference (no API costs)
- **Persistent** - FAISS index + plots survive restarts
- **Privacy-First** - Complete mode isolation, no data cross-contamination

## 🧠 Three Modes of Intelligence

### 📊 **Data Analysis Mode**
- Upload CSV/Excel files for direct analysis
- Generate plots: "line plot sales vs date" → instant visualization
- Ask questions about your data
- Statistics, summaries, and data exploration

### 📚 **Document Search Mode**
- Upload PDFs, TXT files, or images
- Semantic search through document content
- Get answers from your knowledge base
- Source attribution for transparency

### 🌐 **Internet Research Mode**
- Answer questions using general knowledge
- No uploads needed
- Perfect for research and information gathering

## 🛠️ Tech Stack
Streamlit - FAISS - Llama3 - Docker - pandas - sentence-transformers - matplotlib - Ollama

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Ollama installed and running

### Installation
```bash
pip install -r requirements.txt
```

### Run (3 Terminals)

**Terminal 1 - Start Ollama:**
```bash
ollama serve
```

**Terminal 2 - Start App Server:**
```bash
streamlit run app.py --server.port 8502 --server.headless true
```

**Terminal 3 - Start UI:**
```bash
streamlit run agent_ui.py --server.port 8501
```

**45 seconds later → http://localhost:8501 = READY!** 🎉

## 📊 Usage Examples

### Data Analysis Mode
1. Select "📊 Data Analysis" from sidebar
2. Upload your CSV/Excel file
3. Ask questions:
   - "create a bar chart for revenue by month"
   - "show me the statistics"
   - "scatter plot of age vs income"

### Document Search Mode
1. Select "📚 Document Search" from sidebar
2. Upload PDF, TXT, or image files
3. Search your documents:
   - "what are the main findings?"
   - "summarize the methodology"
   - "find information about cost analysis"

### Internet Research Mode
1. Select "🌐 Internet Research" from sidebar
2. No uploads needed
3. Ask any question:
   - "latest AI trends"
   - "how do neural networks work?"
   - "explain quantum computing"

## 🏗️ Architecture

### Mode Isolation
Each mode operates independently:
- **Data Mode** → Only accesses uploaded CSV/Excel files
- **Document Mode** → Only searches FAISS knowledge base
- **Web Mode** → Pure LLM, no external data access

### Data Flow
```
User Input
    ↓
Mode Selector (Sidebar)
    ↓
    ├→ Data Analysis Handler (file-based)
    ├→ Document Search Handler (FAISS-based)
    └→ Internet Research Handler (LLM-based)
    ↓
Response + Visualizations
```

## 📁 Project Structure
```
project/
├── app.py                     # Core ML functions (FAISS, analysis, plotting)
├── agent_ui.py                # Streamlit interface (3 modes)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── faiss_index/               # Knowledge base (auto-created)
│   ├── index.faiss
│   └── index.pkl
└── analysis_plots/            # Generated visualizations (auto-created)
    └── plot_*.png
```

## 🎨 Supported Visualizations

### Plot Types
- **Line Plot** - Trends over time
- **Bar Chart** - Categorical comparisons
- **Scatter Plot** - Correlations
- **Histogram** - Distributions
- **Box Plot** - Statistical summaries

### Natural Language Examples
```
"line plot sales vs date"
"bar chart revenue by product"
"scatter plot age vs income"
"histogram of prices"
"box plot for each category"
```

## ⚙️ Configuration

### Change LLM Model
Edit `app.py`:
```python
llm = ChatOllama(model="llama2")  # or mistral, neural-chat, etc.
```

### Adjust FAISS Chunking
For better search precision, modify chunk size:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Larger = fewer, broader chunks
    chunk_overlap=300,
)
```

### Change Server Ports
```bash
streamlit run agent_ui.py --server.port 8503
```

## 🔒 Privacy & Security
- ✅ **Mode Isolation** - No cross-contamination between modes
- ✅ **Knowledge Base Privacy** - Users can't view entire FAISS index
- ✅ **Temporary Files** - Cleaned up automatically
- ✅ **Local Processing** - No data sent to external APIs (using Ollama locally)
- ✅ **Source Attribution** - Always shows where information comes from

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'mcp'"
```bash
pip install --break-system-packages mcp langchain
```

### Ollama connection refused
```bash
# Make sure Ollama is running in Terminal 1
ollama serve
```

### FAISS index not found
```bash
# Index is auto-created on first document upload
# Just upload a PDF in Document Search mode
```

### Plots not generating
- Verify column names in your CSV match your query
- Check data types (numeric columns required for plots)
- Example: "line plot sales vs date" - both columns must exist

### Streamlit port already in use
```bash
streamlit run agent_ui.py --server.port 8503
```

## 📈 Performance Tips
1. **Chunk Size** - Smaller chunks (500-800) = slower but more precise
2. **Ollama Model** - Larger models = slower but better answers
3. **FAISS Index** - Grows with documents, rebuilds on upload
4. **Data Size** - CSV files <100MB recommended for smooth performance

## 🚀 Production Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "agent_ui.py", "--server.port", "8501"]
```

### Deploy
```bash
docker build -t rag-analyst .
docker run -p 8501:8501 rag-analyst
```

## 📚 Example Workflows

### Financial Analysis Workflow
1. Upload quarterly_data.xlsx (Data Mode)
2. "show me stats for revenue"
3. "line plot revenue vs quarters"
4. "what's the average profit margin?"

### Research Workflow
1. Upload research_papers.pdf (Document Mode)
2. "summarize the key findings"
3. "what methodology was used?"
4. "find limitations discussed"

### Knowledge Gathering Workflow
1. Select Internet Research mode
2. "explain blockchain technology"
3. "latest developments in AI 2024"
4. "pros and cons of machine learning"

## 🤝 Contributing
Issues and pull requests welcome!

## 📄 License
MIT License

## 🎯 Roadmap
- [ ] Web search integration for Internet Research mode
- [ ] Multi-file upload support for Data Analysis
- [ ] Custom chart templates and themes
- [ ] Document filtering by date/source
- [ ] Query history and analytics per mode
- [ ] Export reports to PDF
- [ ] Collaborative analysis features

---

**Built for Data Scientists** - Your conversational data analyst! 🚀

**Need help?** Check troubleshooting section or open an issue.