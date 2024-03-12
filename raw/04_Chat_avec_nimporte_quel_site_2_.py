# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_openai import OpenAI

load_dotenv()

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store, document_chunks  # Return both the vector store and document chunks

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Compte tenu de la conversation ci-dessus, générer une requête de recherche pour trouver des informations pertinentes pour la conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Répondez aux questions de l'utilisateur en vous basant sur le contexte ci-dessous :\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    # Assuming retriever_chain can be modified to include or return intermediary steps
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response, intermediary_steps = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    }, return_intermediary_steps=True)  # Hypothetical parameter to get steps
    return response['answer'], intermediary_steps

def generate_questions_from_content(document_chunks):
    questions = []
    llm = OpenAI()
    for chunk in document_chunks[:3]:  # Limit to first 3 chunks for demonstration
        prompt = f"Given the following text, generate a relevant question in French :\n\n{chunk}"
        question = llm(prompt)
        questions.append(question)
    return questions

# Streamlit page configuration
st.set_page_config(page_title="Discuter avec des sites web", page_icon="🤖")
st.title("Discuter avec des sites web 🌐")
st.markdown("""
**Bienvenue sur notre assistant search connecté au web**, votre outil privilégié pour interagir avec des sites : les blogs de vos concurrents, ou encore des sites officiels comme Ameli...
""")

# Sidebar for user input
with st.sidebar:
    st.header("Paramètres")
    website_url = st.text_input("URL du site web", help="Entrez l'URL du site web avec lequel vous souhaitez interagir.")

if not website_url:
    st.info("Veuillez entrer une URL de site web pour commencer.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Bonjour, je suis le bot Search Assistant de Seyna AI. Comment puis-je vous aider ?")]

    if "vector_store" not in st.session_state:
        vector_store, document_chunks = get_vectorstore_from_url(website_url)
        st.session_state.vector_store = vector_store
        questions = generate_questions_from_content(document_chunks)
        st.session_state.questions = questions
        if questions:
            st.write("Questions Generated from the Website:")
            for question in questions:
                st.write(question)

    user_query = st.chat_input("Tapez votre message ici...")
    if user_query:
        answer, intermediary_steps = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("IA"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Humain"):
                    st.write(message.content)

        # Optionally, display intermediary steps
        if intermediary_steps:  # Assuming intermediary_steps is a list of steps
            with st.expander("See how I thought this through"):
                for step in intermediary_steps:
                    st.json(step)  # Assuming step is in a format that st.json can render nicely
