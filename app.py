
import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import streamlit as st
import requests
import logging

# Setup custom logger for tools only
tool_logger = logging.getLogger("tool_logger")
tool_logger.setLevel(logging.INFO)
tool_logger.propagate = False  

if not tool_logger.handlers:
    handler = logging.FileHandler('agent_decisions.log', mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tool_logger.addHandler(handler)

logging.getLogger().setLevel(logging.ERROR)  

if 'logger_configured' not in st.session_state:
    st.session_state.logger_configured = True

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

system_prompt = """You are a question-answering assistant. 
Answer user queries in a polite and professional manner. 
If the user mentions 'calculate', 'find meaning', etc., use your given set of tools to accomplish the task."""

def dictionary_lookup(word: str) -> str:
    tool_logger.info(f"Tool Called: dictionary_lookup with word='{word}'")
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        definitions = [entry["meanings"][0]["definitions"][0]["definition"] for entry in data if "meanings" in entry]
        return f"Definition of '{word}': {definitions}"
    else:
        return f"Could not fetch definition for '{word}'. Please try another word."

def calculator(expression: str) -> str:
    tool_logger.info(f"Tool Called: calculator with expression='{expression}'")
    return str(eval(expression))

# RAG setup
knowledge_folder = "./knowledge"
loader = DirectoryLoader(
    path=knowledge_folder,
    glob="**/*.txt",
    loader_cls=TextLoader
)
pdf_loader = DirectoryLoader(
    path=knowledge_folder,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs = loader.load() + pdf_loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = FAISS.from_documents(documents, hf_embeddings)
retriever = vector.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 documents

# Custom retriever function with logging
def retrieve_documents(query: str) -> list:
    tool_logger.info(f"Tool Called: KnowledgeBaseRetriever with query='{query}'")
    docs = retriever.get_relevant_documents(query)
    return docs  # Return list of Document objects

retriever_tool = Tool(
    name="KnowledgeBaseRetriever",
    func=retrieve_documents,
    description="Query answering tool, always use it when no specific tool is mentioned."
)

# Define tools
dictionary_tool = Tool(
    name="dictionary_lookup",
    func=dictionary_lookup,
    description="Fetches word definitions. Only input a single word."
)
calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="Performs mathematical calculations."
)
tools = [dictionary_tool, calculator_tool, retriever_tool]

# Setup agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=True)

# Streamlit UI
st.set_page_config(page_title="RAG Q&A Assistant", layout="wide")
st.title("Knowledge Assistant")

# Sidebar for logs
with st.sidebar:
    st.header("Agent Decision Logs")
    log_placeholder = st.empty()
    try:
        with open('agent_decisions.log', 'r') as log_file:
            logs = log_file.readlines()
            log_placeholder.text("\n".join([log.strip() for log in logs[-10:]]))  # Show last 10 logs
    except FileNotFoundError:
        log_placeholder.text("No logs available yet.")

    # Display knowledge base files
    st.header("Knowledge Base Files")
    try:
        knowledge_files = [f for f in os.listdir(knowledge_folder) if f.endswith(('.txt', '.pdf'))]
        if knowledge_files:
            st.markdown("**Loaded Files:**")
            for file in knowledge_files:
                st.write(f"- {file}")
        else:
            st.write("No text or PDF files found in ./knowledge directory.")
    except FileNotFoundError:
        st.write("Knowledge directory not found. Please create ./knowledge and add files.")

# Main input and output
user_input = st.text_input("Ask me anything", "")
if user_input:
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        answer = response.get("output", "No answer generated.")
        st.success(f"**Answer**: {answer}")
        for step in response.get("intermediate_steps", []):
            if step[0].tool == "KnowledgeBaseRetriever":
                st.markdown("### Retrieved Chunks")

                
                for idx, doc in enumerate(step[1]):
                    with st.expander(f"Chunk {idx + 1}"):
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content)
                        st.write(f"Debug: Retrieved {len(step[1])} chunks from retriever tool.")
                break  
        try:
            with open('agent_decisions.log', 'r') as log_file:
                logs = log_file.readlines()
                log_placeholder.text("\n".join([log.strip() for log in logs[-10:]]))  
        except FileNotFoundError:
            log_placeholder.text("No logs available yet.")