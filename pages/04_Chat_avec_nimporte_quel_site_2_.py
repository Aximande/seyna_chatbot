import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=OPENAI_API_KEY))
    return vector_store

def get_context_retriever_chain(vector_store):
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
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Répondez aux questions de l'utilisateur en vous basant sur le contexte ci-dessous :\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# Nouvelle fonction pour générer des questions pertinentes
def generate_questions(vector_store):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    context = " ".join([doc["text"] for doc in vector_store.documents[:10]])  # Utiliser les premiers documents comme contexte
    prompt = f"Générez trois questions pertinentes basées sur le contenu suivant :\n\n{context}"
    generated_questions = llm(prompt=prompt, max_tokens=100)
    questions = generated_questions.split("\n")[:3]  # Assurez-vous que la sortie est traitée correctement pour obtenir une liste de questions
    return questions

# Configuration de la page Streamlit
st.set_page_config(page_title="Discuter avec des sites web", page_icon="🤖")
st.title("Discuter avec des sites web 🌐")

# Barre latérale
with st.sidebar:
    st.header("Paramètres")
    website_url = st.text_input("URL du site web", help="Entrez l'URL du site web avec lequel vous souhaitez interagir.")

if not website_url:
    st.info("Veuillez entrer une URL de site web pour commencer.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Bonjour, je suis un bot. Comment puis-je vous aider ?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        st.session_state.generated_questions = generate_questions(st.session_state.vector_store)  # Générer des questions après le chargement du vector_store

    # Afficher les questions générées sous forme de boutons
    st.header("Questions suggérées")
    for question in st.session_state.generated_questions:
        if st.button(question):
            st.session
