import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables (API Keys)
load_dotenv()

# 1. Page Configuration
st.set_page_config(page_title="AI Engineering Assistant", page_icon="🛠", layout="wide")
st.title("🛠 AI Assistant for Datasheets & Manuals")
st.caption("Your wise embedded systems mentor with a touch of engineering humor")

# Ensure API Key is available
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OPENAI_API_KEY! Please check your .env file.")
    st.stop()

# 2. Cached Vector Database Initialization
@st.cache_resource
def init_rag():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    if not docs:
        return None
        
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="vector_db")
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = init_rag()

# Sidebar Document Status
with st.sidebar:
    st.header("📦 Document Status")
    if retriever:
        st.success("Technical documentation indexed successfully!")
    else:
        st.warning("The 'data/' folder is empty. Please add your PDF datasheets inside it.")

# 3. LLM Pipeline Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an experienced electronics engineer, a senior embedded systems developer, "
        "and a wise mentor with a subtle sense of professional humor. Your goal is to help "
        "users understand datasheets and hardware reference manuals.\n\n"
        "Use the following technical documentation context to answer the user's question. "
        "If the information cannot be found in the provided context, state honestly that "
        "it is not present in the given datasheet.\n\n"
        "Datasheet Context:\n{context}\n"
    )),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])
chain = prompt | llm | StrOutputParser()

# 4. Streamlit Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input and Interaction
if user_query := st.chat_input("Ask a question about your datasheet..."):
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve context from DB
    context_text = ""
    if retriever:
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([
            f"[Page {d.metadata.get('page', 'Unknown')}]: {d.page_content}" 
            for d in retrieved_docs
        ])

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Invoke LangChain pipeline
        response = chain.invoke({
            "context": context_text, 
            "input": user_query, 
            "chat_history": []
        })
        
        response_placeholder.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
