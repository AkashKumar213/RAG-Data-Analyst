# 🤖 TriSense AI

A Streamlit-based application for:

* 📊 Data analysis (CSV/Excel)
* 📚 Document search (FAISS-based RAG)
* 🌐 Hybrid research (LLM + optional document retrieval)

---

## 🚀 What It Does

* Convert natural language into charts and insights
  → `"line plot sales vs date"`

* Search across uploaded documents (PDF, TXT, images)

* Perform research using LLM, with optional knowledge base support

* Maintain chat history with auto-save support

---

## 🛠️ Tech Stack

* Streamlit
* FAISS
* Ollama (Llama3)
* pandas
* matplotlib

---

## ⚡ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Start Ollama

```bash
ollama serve
```

---

### 3. Run backend

```bash
streamlit run app.py --server.port 8502 --server.headless true
```

---

### 4. Run UI

```bash
streamlit run agent_ui.py --server.port 8501
```

---

### 5. Open in browser

```
http://localhost:8501
```

---

## 📊 Usage

### Data Mode

* Upload CSV/Excel
* Ask:

```
line plot revenue vs date
show summary statistics
```

---

### Document Mode

* Upload files
* Ask:

```
summarize the document
find key points
```

---

### Research Mode

* Ask any question
* Automatically uses documents if available

---

## 📁 Project Structure

```
project/
├── app.py
├── agent_ui.py
├── requirements.txt
├── faiss_index/
└── analysis_plots/
```

---

## 🐛 Troubleshooting

**Ollama not running**

```bash
ollama serve
```

**Port in use**

```bash
streamlit run agent_ui.py --server.port 8503
```

**FAISS not found**

* Upload a document to initialize

---

## 🗺️ Roadmap

* [ ] Voice interaction
* [ ] Chat search
* [ ] Multi-dataset support
* [ ] Source citations
* [ ] Vector DB options (FAISS / Chroma / PGVector)
* [ ] Export chats

---

## 📄 License

MIT
