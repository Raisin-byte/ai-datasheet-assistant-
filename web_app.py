import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Настройка страницы Streamlit
st.set_page_config(page_title="AI Engineering Assistant", page_icon="🛠️", layout="wide")
st.title("🛠️ AI Assistant for Datasheets & Manuals")
st.caption("Твой мудрый инженерный ментор с тонким юмором")

# Загружаем ключи
load_dotenv()

# Кэшируем загрузку базы данных, чтобы она не пересоздавалась при каждом клике
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

# Выводим статус загрузки файлов в боковую панель
with st.sidebar:
    st.header("📦 Статус документов")
    if retriever:
        st.success("Техническая документация успешно проиндексирована!")
    else:
        st.warning("Папка data/ пуста. Закинь туда PDF-файлы на компьютере.")

# 2. Инициализация ИИ
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Ты — опытный инженер-электронщик, senior-разработчик встроенных систем (Embedded) "
        "и мудрый учитель с тонким юмором. Твоя цель — помогать разбираться в даташитах.\n\n"
        "Используй следующий контекст из технической документации, чтобы ответить на вопрос пользователя. "
        "Если информации нет в контексте, честно скажи, что в предоставленном даташите этого не нашел.\n\n"
        "Контекст из даташита:\n{context}\n"
    )),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])
chain = prompt | llm | StrOutputParser()

# 3. История чата в интерфейсе
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем старые сообщения
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Поле ввода для пользователя
if user_query := st.chat_input("Задай вопрос по даташиту..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Поиск контекста
    context_text = ""
    if retriever:
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([f"[Стр. {d.metadata.get('page', 'Неизвестно')}]: {d.page_content}" for d in retrieved_docs])

    # Запрос к ИИ
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # Собираем историю для промпта (в упрощенном виде для теста)
        response = chain.invoke({"context": context_text, "input": user_query, "chat_history": []})
        response_placeholder.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
