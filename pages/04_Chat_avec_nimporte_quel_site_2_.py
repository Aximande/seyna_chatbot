import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialisez chat_history au début pour s'assurer qu'il existe avant toute utilisation
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def get_vectorstore_and_text_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks, original_texts = text_splitter.split_documents(document, return_originals=True)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=OPENAI_API_KEY))
    full_text = " ".join(original_texts)  # Combine les textes originaux pour la génération de questions
    return vector_store, full_text

def generate_questions(full_text):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Générez trois questions pertinentes basées sur le contenu suivant :\n\n{full_text[:2000]}"  # Utilisez une partie du texte pour éviter de dépasser la limite de tokens
    generated_questions = llm.generate(prompt=prompt, max_tokens=100, n=3, stop=["\n"], temperature=1.0)
    questions = [q["choices"][0]["text"].strip() for q in generated_questions.responses]  # Adaptez selon la structure de réponse de l'API
    return questions

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


# Configuration de la page Streamlit
st.set_page_config(page_title="Discuter avec des sites web", page_icon="🤖")
st.title("Discuter avec des sites web 🌐")

# Barre latérale
with st.sidebar:
    st.header("Paramètres")
    website_url = st.text_input("URL du site web", help="Entrez l'URL du site web avec lequel vous souhaitez interagir.")

if website_url:
    if "vector_store" not in st.session_state or website_url != st.session_state.get('last_website_url', ''):
        st.session_state.vector_store, full_text = get_vectorstore_and_text_from_url(website_url)
        st.session_state.generated_questions = generate_questions(full_text)
        st.session_state.last_website_url = website_url  # Gardez une trace de l'URL pour éviter de recharger inutilement
        for question in st.session_state.generated_questions:
            if st.button(question):
                user_query = question  # Simulez une entrée utilisateur avec la question générée
                response = get_response(user_query)
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.info("Veuillez entrer une URL de site web pour commencer.")

def get_response(user_input):
    """
    Obtient une réponse de l'agent LLM basée sur l'entrée de l'utilisateur et l'historique de la conversation.
    """
    # Assurez-vous que vector_store et chat_history sont initialisés dans st.session_state
    if 'vector_store' not in st.session_state or 'chat_history' not in st.session_state:
        return "Veuillez charger un site web pour commencer la conversation."

    # Récupère le chain de rétrieval contextuel pour le vector_store actuel
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    # Crée le chain de conversation basé sur le chain de rétrieval
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Invoque le chain de conversation avec l'entrée utilisateur et l'historique de la conversation
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    # Ajoute la réponse à l'historique de la conversation
    st.session_state.chat_history.append(AIMessage(content=response))

    return response

def get_context_retriever_chain(vector_store):
    """
    Crée un chain de rétrieval contextuel pour une instance de Chroma vector_store.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Compte tenu de la conversation ci-dessus, générer une requête de recherche pour trouver des informations pertinentes pour la conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    """
    Crée un chain de conversation RAG (Retrieval-Augmented Generation) basé sur un retriever_chain.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Répondez aux questions de l'utilisateur en vous basant sur le contexte ci-dessous :\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Logique pour afficher les messages de l'IA et de l'utilisateur
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("IA"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Humain"):
            st.write(message.content)
