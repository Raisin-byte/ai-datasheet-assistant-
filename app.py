import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Автоматически подтягиваем ваш рабочий ключ из файла .env
load_dotenv()

# 2. Инициализируем модель ChatGPT (активируем 10 долларов)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. Задаем роль для нашего ИИ-помощника
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — опытный инженер-электронщик, senior-разработчик встроенных систем (Embedded) "
        "и по совместительству мудрый, поддерживающий учитель. Твоя цель — помогать "
        "пользователю разбираться в сложных даташитах, архитектуре чипов и коде.\n\n"
        "Правила общения:\n"
        "1. Отвечай строго профессионально, глубоко и технически грамотно.\n"
        "2. Разбавляй ответы тонким, добрым инженерным юмором (без токсичности и жесткого сарказма).\n"
        "3. Объясняй сложные концепции простыми словами, как лучший университетский ментор.\n"
        "4. Общайся на русском и английском языках."),
    ("user", "{input}")
])
# Дополнительные импорты для работы с PDF и ChromaDB 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Проверяем, что папка data существует
if not os.path.exists("data"):
    os.makedirs("data")

print("1. Загрузка PDF-файлов из папки data/...")
loader = PyPDFDirectoryLoader("data")
docs = loader.load()

if not docs:
    print("❌ Папка data/ пуста! Закинь туда хотя бы один PDF-даташит.")
else:
    print(f"✅ Успешно загружено страниц из PDF: {len(docs)}")

# Инициализируем переводчик текста в векторы
embeddings = OpenAIEmbeddings()

print("2. Индексация документов в базу данных Chroma...")
vector_store = Chroma.from_documents(docs, embeddings, persist_directory="vector_db") if docs else None
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

# Собираем конвейер
chain = prompt | llm | StrOutputParser()

# Запуск диалога в консоли
print("\n🚀 Ассистент готов к работе!")
while True:
    user_query = input("\nАрсений (или 'выход'): ")
    if user_query.lower() in ['выход', 'quit', 'exit']:
        break
        
    context_text = ""
    if retriever:
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([f"[Стр. {d.metadata.get('page', 'Неизвестно')}]: {d.page_content}" for d in retrieved_docs])
    
    response = chain.invoke({"context": context_text, "input": user_query})
    print(f"\nИИ-Инженер:\n{response}")
