# 🧠 Knowledge Agent

## Overview
This project implements a single knowledge agent with three tools: a retrieval tool, a calculator tool, and a dictionary tool. Built as part of an internship assignment, the agent processes user queries by routing them to the appropriate tool and generates natural-language answers using a large language model (LLM). It features a Streamlit web interface for user interaction and logs tool decisions in a sidebar.

> 💡 **LLM Note**: This project uses **Google Gemini (gemini-2.0-flash)** via LangChain's `langchain_google_genai` integration. However, **LangChain is highly modular**, so you can easily swap in other LLMs like OpenAI, Cohere, Anthropic, etc., with minor changes.

---

## Architecture

The system is composed of the following components:

1. **Data Ingestion**:
   - **Input**: Text and PDF files in the `./knowledge` directory (e.g., company FAQs, product specs).
   - **Process**: Uses `langchain_community.document_loaders` (`DirectoryLoader`, `TextLoader`, `PyPDFLoader`) to load documents. Documents are split into chunks (1000 characters, 200 overlap) using `RecursiveCharacterTextSplitter`.
   - **Output**: Chunked documents ready for vector indexing.

2. **Vector Store & Retrieval**:
   - **Vector Store**: FAISS (`langchain_community.vectorstores.FAISS`) with HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
   - **Retrieval**: A FAISS retriever fetches the top 3 relevant document chunks for a query (`search_kwargs={"k": 3}`).
   - **Tool**: A custom `Tool` (`KnowledgeBaseRetriever`) handles retrieval and logs actions.

3. **LLM Integration**:
   - **Model**: Google’s `gemini-2.0-flash` via `langchain_google_genai`.
   - **Role**: Generates natural-language answers based on retrieved context or tool outputs.

4. **Agentic Workflow**:
   - **Framework**: LangChain’s `create_tool_calling_agent` and `AgentExecutor`.
   - **Tools**:
     - `dictionary_lookup`: Fetches word definitions using the Dictionary API.
     - `calculator`: Evaluates mathematical expressions.
     - `KnowledgeBaseRetriever`: Retrieves relevant document chunks.
   - **Routing Logic**:
     - Queries with “calculate” route to the calculator tool.
     - Queries with “find meaning” or similar route to the dictionary tool.
     - Other queries default to the RAG pipeline (retriever → LLM).
   - **Logging**: Tool calls and retrievals are logged to `agent_decisions.log` using a custom logger.

5. **Demo Interface**:
   - **Framework**: Streamlit (`streamlit`).
   - **Features**:
     - Text input for user queries.
     - Display of answers.
     - Sidebar showing the last 10 log entries (tool calls and retrievals).
   - **Implementation**: Updates logs reactively after each query.

---

## 📁 Folder Structure

```
├── knowledge/              # Place your .txt and .pdf knowledge files here
├── agent_decisions.log     # Auto-generated log file for tool usage
├── .env                    # Your environment variables (API keys)
├── requirements.txt        # All Python dependencies
├── app.py                  # Main application file (Streamlit)
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant
```

### 2. Add Knowledge Files

Place your `.txt` and `.pdf` documents inside the `knowledge/` folder.  
Dummy documents are already included for testing.

### 3. Create a `.env` File

In the root directory, create a `.env` file and add your [Google Generative AI API Key](https://makersuite.google.com/app/apikey):

```
GOOGLE_API_KEY=your_google_api_key_here
```

> 🔐 This key is required for using Gemini via LangChain. You can replace Gemini with other LLMs if desired by modifying the code slightly.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📓 Usage

- Type a query like: `What is AI?` or `define entropy` or `calculate 23 * 45`
- The assistant will use:
  - Knowledge Base RAG for general queries
  - Dictionary API for definitions
  - Python `eval()` for calculations
- Logs will appear in the sidebar and also be saved in `agent_decisions.log`.

---

## 🛠 Tech Stack

- **LangChain** (modular agent + tool framework)
- **Google Generative AI (Gemini 2.0)** (default LLM)
- **FAISS Vector Store**
- **HuggingFace Embeddings**
- **Streamlit**
- **Python Logging**

---

