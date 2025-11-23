# %%time
import streamlit as st

gauth = st.secrets["gigachat_api_key"]
dauth = st.secrets["deepseek_api_key"]

from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(
    model="deepseek/deepseek-chat-v3.1",
    api_key=dauth,
    api_base='https://api.proxyapi.ru/openrouter/v1',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

auth = gauth
embeddings = GigaChatEmbeddings(
    credentials=gauth,
    verify_ssl_certs=False
)

import os.path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

db_file_name = "db_pdf/db_01"
file_path = db_file_name + "/index.faiss"

if os.path.exists(file_path):
    db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)

embedding_retriever = db.as_retriever(search_kwargs={"k": 10})

# question = "Цетровка"
def generate_response(question):
    
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    
    template = """
    Ты — полезный помощник для вопросно ответных приложений. Используй следующий контекст {context},
    чтобы ответить на вопрос {question}.
    Если ответа нет в контексте — скажи, что не знаешь.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | StrOutputParser()
    
    context = embedding_retriever.invoke(question)
    response = chain.invoke({"context": context, "question": question})

    return response

result = ""

with st.form(key='qa_form', clear_on_submit=True, border=True):
    st.subheader('Основы метеорологии')
    query_text = st.text_input(
    'Отправьте свой вопрос LLM:',
    placeholder='Здесь нужно написать вопрос',
)
    submitted = st.form_submit_button("Отправить")

    if submitted:
        with st.spinner('Calculating...'):
            # Генерируем ответ с помощью функции
            response = generate_response(query_text)
            result = response

# Отображаем результат, если он есть
if result:
    st.info(result)
    
