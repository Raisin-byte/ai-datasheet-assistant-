import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Настройка страницы Streamlit (красивый интерфейс ChatGPT)
st.set_page_config(page_title="AI Engineering Assistant", page_icon="🛠️", layout="wide")
st.title("🛠️ AI Assistant for Datasheets & Manuals")
st.caption("Твой мудрый инженерный ментор с тонким юмором")

# Прописываем твой рабочий ключ OpenAI намертво
API_KEY = "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" # ПОДСТАВЬ СЮДА СВОЙ ПОЛНЫЙ КЛЮЧ OpenAI!

BASE_DIR = os.getcwd()
pdf_path = os.path.join(BASE_DIR, "data", "lecture 3.pdf") 

@st.cache_resource
def init_rag():
    if not os.path.exists(pdf_path):
        return None
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            return None
        embeddings = OpenAIEmbeddings(api_key= API_KEY)
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=os.path.join(BASE_DIR, "vector_db"))
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        return None

retriever = init_rag()

# Красивая боковая панель статуса
with st.sidebar:
    st.header("📦 Статус документов")
    if retriever:
        st.success("Лекция 'lecture 3.pdf' успешно найдена и прочитана!")
    else:
        st.error("Файл 'lecture 3.pdf' не найден!")
        st.info(f"Убедись, что файл лежит по пути: {pdf_path}")

# Инициализация ИИ
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — опытный инженер-электронщик, senior-разработчик встроенных систем (Embedded) "
        "и мудрый учитель с тонким юмором. Твоя цель — помогать разбираться в даташитах.\n\n"
        "Используй следующий контекст из технической документации, чтобы ответить на вопрос пользователя:\n{context}"
    )),
    ("user", "{input}")
])
chain = prompt | llm | StrOutputParser()

# Сохранение истории чата в стиле Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода, как в настоящем ChatGPT
if user_query := st.chat_input("Задай вопрос по даташиту..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    context_text = ""
    if retriever:
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([f"[Стр. {d.metadata.get('page', 'Неизвестно')}]: {d.page_content}" for d in retrieved_docs])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response = chain.invoke({"context": context_text, "input": user_query})
        response_placeholder.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})