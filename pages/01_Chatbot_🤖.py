from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
import streamlit as st
import os
from PIL import Image

from langchain.schema.messages import SystemMessage

from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

jobs = {
    "Gestion de sinistre": ["Sinistre 1", "Sinistre 2", "Sinistre 3"],
    "Assistant sales": ["Assistant 1", "Assistant 2", "Assistant 3"],
}

templates = {
    "Gestion de sinistre": {
        "Sinistre 1": (
            "En tant qu'assistant spécialisé dans la gestion de sinistres, "
            "je suis là pour vous aider à naviguer dans le processus de "
            "déclaration et de suivi de votre sinistre. Je peux vous fournir "
            "des informations sur les étapes à suivre, les documents nécessaires "
            "et comment assurer un suivi efficace de votre dossier."
        ),
        "Sinistre 2": (
    "En tant qu'assistant spécialisé en gestion de sinistres liés aux loyers impayés, je possède une expertise approfondie "
    "dans les procédures de recouvrement. Voici comment je peux vous assister : \n\n"
    "- **Glossaire** : Je maîtrise le vocabulaire spécifique tel que ADB (Administrateur De Bien), CDP (Commandement De Payer), "
    "CDQL (Commandement De Quitter les Lieux), et MED (Mise En Demeure), essentiels pour naviguer dans les procédures de recouvrement.\n\n"
    "- **Procédure de recouvrement** : Je vous guide pas à pas à travers le processus, depuis l'envoi de la Mise En Demeure par l'ADB "
    "jusqu'à la réquisition de la force publique si nécessaire. Cela inclut l'envoi du commandement de payer après deux mois de dette, "
    "la déclaration à Gme dans les délais pour éviter l'application de la carence, et l'accompagnement pour l'assignation en justice.\n\n"
    "- **Support Administratif** : Je fournis un soutien pour l'envoi mensuel d'extraits de compte actualisés du locataire et assure un suivi "
    "du règlement après réception, avec le premier règlement effectué dans les cinq jours ouvrés après une déclaration complète de l'ADB.\n\n"
    "Avec ma connaissance approfondie du domaine et mon expertise en procédures administratives, je suis ici pour vous aider à naviguer dans "
    "les complexités des sinistres liés aux loyers impayés et optimiser votre processus de recouvrement."
),

        "Sinistre 3": (
            "Spécialisé dans les sinistres liés à l'habitation, je suis ici pour répondre "
            "à toutes vos questions concernant la déclaration de sinistre, la couverture d'assurance "
            "en cas de dégâts des eaux, incendie ou vol, et le processus de réclamation."
        ),
    },
    "Assistant sales": {
        "Assistant 1": (
            "Je suis votre copilote de vente SeynaGPT, prêt à booster vos performances de vente en assurance. "
            "Je vous fournirai des insights sur les meilleures pratiques de vente, comment identifier les besoins "
            "des clients et proposer des solutions d'assurance adaptées."
        ),
        "Assistant 2": (
            "Avec une expertise en vente d'assurances santé, je suis là pour vous aider à comprendre les différents plans "
            "d'assurance santé, leurs avantages, et comment les vendre efficacement en mettant en avant leurs valeurs ajoutées pour les clients."
        ),
        "Assistant 3": (
            "Spécialisé dans les assurances vie, je vous accompagne pour maîtriser les arguments de vente, comprendre les préoccupations "
            "des clients concernant la sécurité financière de leur famille et proposer des polices d'assurance vie qui répondent à leurs besoins spécifiques."
        ),
    },
}


def prepare_file(uploaded_file):
    path = None
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def agent_without_rag(template):
    # LLM

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation


def agent_rag(loader, template, doc_type="PDF"):
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_in_document",
        "Searches and returns documents.",
    )
    tools = [tool]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    context = (
        template
        + """

    Your task will be to complete the request of the user and using the provided {doc_type} by the user.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise

    Remember it's very important your answer should always be in French

    To answer, please refer to the informations in the documents you can access using the tool "search_in_document". Never ask the user for the document. The document is already given. Use the function "search_in_document".
    """
    )
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm,
        tools,
        system_message=sys_message,
        verbose=True,
    )

    return agent_executor


def query(agent, question):
    with st.spinner("Waiting for response..."):
        response = agent.invoke({"input": question})
        if "text" in response:
            response = response["text"]
        else:
            response = response["output"]
    return response


def update_agent(file):
    def conf_changed():
        return (
            st.session_state.previous_agent["category"] != st.session_state.categorie
            or st.session_state.previous_agent["job"] != st.session_state.tache
            or st.session_state.previous_agent["document"] != st.session_state.document
        )

    def file_changed():
        return ("file" in st.session_state and st.session_state.file != file) or (
            "file" not in st.session_state and file is not None
        )

    if "agent" not in st.session_state or conf_changed() or file_changed():
        st.session_state.messages = []
        st.session_state.previous_agent["category"] = st.session_state.categorie
        st.session_state.previous_agent["job"] = st.session_state.tache
        st.session_state.previous_agent["document"] = st.session_state.document
        st.session_state.file = file

        with st.spinner("Preparing agent..."):
            if st.session_state.document == "PDF":
                if file is None:
                    st.session_state.agent = None
                else:
                    file_path = prepare_file(file)
                    loader = PyPDFLoader(file_path)
                    st.session_state.agent = agent_rag(
                        loader,
                        templates[st.session_state.categorie][st.session_state.tache],
                        "PDF",
                    )
            elif st.session_state.document == "CSV":
                if file is None:
                    st.session_state.agent = None
                else:
                    file_path = prepare_file(file)
                    loader = CSVLoader(file_path)
                    st.session_state.agent = agent_rag(
                        loader,
                        templates[st.session_state.categorie][st.session_state.tache],
                        "CSV",
                    )
            else:
                st.session_state.agent = agent_without_rag(
                    templates[st.session_state.categorie][st.session_state.tache]
                )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.previous_agent = {
        "category": "Gestion de sinistre",
        "job": "Sinistre 1",
        "document": "Aucun",
    }

st.set_page_config(page_title="Assistant chatbot")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image(
    Image.open("static/logo_seyna_150_orange.png"),
    width=300,
)
st.title("Chatbot 🤖")

st.radio(
    "Choisissez votre categorie :",
    (
        "Gestion de sinistre",
        "Assistant sales",
    ),
    key="categorie",
)

# we create a placeholder list to store the jobs in the category

st.radio(
    "Choisissez votre tache :",
    jobs[st.session_state.categorie],
    key="tache",
)

st.radio(
    "Document ?",
    ("Aucun", "PDF", "CSV"),
    key="document",
)

if st.session_state.document == "PDF":
    file = st.file_uploader("Selectionnez le pdf", type="pdf")
elif st.session_state.document == "CSV":
    file = st.file_uploader("Selectionnez le csv", type="csv")
else:
    file = None


update_agent(file)

st.write(templates[st.session_state.categorie][st.session_state.tache])

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

response = ""
# React to user input
if "agent" in st.session_state and st.session_state.agent is not None:
    if prompt := st.chat_input("Another question ?"):
        st.session_state.start = True
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = query(st.session_state.agent, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
