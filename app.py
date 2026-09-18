import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Автоматически подтягиваем ваш рабочий ключ из файла .env
load_dotenv()

# 2. Инициализируем модель ChatGPT (активируем ваши 10 долларов)
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

# 4. Собираем конвейер цепочки (ЭТА СТРОЧКА У ВАС ПРОПАЛА!)
chain = prompt | llm | StrOutputParser()

# 5. Запускаем тест-драйв с нашим вопросом
print("Отправляю запрос по цепочке к OpenAI...")
response = chain.invoke({"input": ""})

# 6. Выводим результат
print("\nОтвет саркастичного ИИ:")
print(response)