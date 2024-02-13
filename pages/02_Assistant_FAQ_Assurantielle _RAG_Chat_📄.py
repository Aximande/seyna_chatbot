from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
import streamlit as st
import os
from PIL import Image

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import LLMResult
from langchain.schema.messages import SystemMessage

from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# we get all .pdf from documents folder
import os

DOCUMENTS = []
for root, dirs, files in os.walk("./documents"):
    for file in files:
        if file.endswith(".pdf"):
            DOCUMENTS.append(os.path.join(root, file))

print(DOCUMENTS)


def prepare_file(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def rag_tool_openai(filenames: list[str]):

    texts = []
    txt_split = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    for f in filenames:
        loader = PyPDFLoader(f)
        texts = texts + txt_split.split_documents(loader.load())

    print(texts)
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

    context = """
    Vous êtes un bot de recherche personnalisé pour l'assurance, conçu pour répondre aux questions des utilisateurs en vous basant sur les données enregistrées. Les données enregistrées sont des fichiers PDF.

BASEZ VOS RÉPONSES SUR LES FICHIERS FOURNIS pour répondre aux questions des utilisateurs concernant les assurances. Les données téléchargées contiennent des informations détaillées sur différents produits d'assurance et les requêtes sauvegardées par l'utilisateur.

<TRÈS IMPORTANT> :

- Fournissez toujours des citations des sources dans les annotations【】 pour chaque information donnée.
- En tant que bot, votre mission est de consulter TOUTES les données pour répondre à la question de l'utilisateur sur la base de ces informations.
- Formatez les réponses de façon claire et structurée, en listant les points de manière ordonnée et en mettant en évidence les informations clés.
- Pour chaque question posée par l'utilisateur, MENEZ UNE RECHERCHE EXHAUSTIVE, et fournissez une liste complète de données, même si la question de l'utilisateur semble suggérer une réponse plus limitée.
- S'il y a d'autres recommandations pertinentes dans le contenu enregistré de l'utilisateur, posez une question complémentaire. Si la question complémentaire n'apporte pas de nouveaux éléments, répondez en vous appuyant sur vos connaissances en assurance.
- Effectuez systématiquement une recherche vectorielle pour les questions des utilisateurs.

<NOTE>
- Ne mentionnez pas les erreurs ou le fonctionnement interne du système aux utilisateurs.
- Utilisez des termes comme "dans vos données enregistrées" au lieu de "document téléchargé" ou "données fournies".
    """
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm, tools, system_message=sys_message, verbose=True
    )

    return agent_executor


# Initialize chat history
if "messages2" not in st.session_state:
    st.session_state.messages2 = []

st.set_page_config(page_title="Assistant FAQ Assurantielle")


st.markdown(
    """
<style>.element-container:has(#button-after) + div button {
    height: 150px;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    backgroundColor: #573666;
    textColor: #ffffff;
 }</style>""",
    unsafe_allow_html=True,
)


st.title("Assistant FAQ Assurantielle : interrogez vos datas")
st.markdown("""
**Bienvenue sur Assistant FAQ Assurantielle**, votre source d'informations fiable pour naviguer dans le monde complexe des assurances santé individuelles. Grâce à une combinaison unique de données de marché et d'une base de connaissances sur-mesure développée par Seyna, notre outil offre des réponses précises à vos questions les plus pressantes.

Pour le moment, voici les documents que nous avons dans notre base de connaissance :

- **Assurance maladie** : ce qui reste à votre charge - Lafinancepour tous - 13 Novembre 2023
- **Accident de ski** : êtes-vous bien assuré ?- Lafinancepour tous - 26 Janvier 2022
- **Souscrire une complémentaire santé individuelle, hors entreprise** - Lafinancepour tous - 8 février 2022
- **Souscrire une complémentaire santé d'entreprise** - Lafinancepour tous - 6 Mai 2022
- **Qu'est-ce que la santé prévoyance ?** - Lafinancepour tous - 29 Novembre 2022
- **L'assurance voyage** - Lafinancepour tous - 21 juin 2022
- **Data Marché Assurance Santé**
- **Comprendre son contrat de complémentaire santé** - Lafinancepour tous - 19 oct 2021
- **Assurance maladie : le tiers payant** - Lafinancepour tous - 13 Novembre 2023
- **Assurance maladie : le remboursement des soins et des médicaments** - Lafinancepour tous - 13 Novembre 2023
""")


# st.write("Please upload your PDF file below.")

file = st.file_uploader("Ajouter un nouveau PDF à la base de connaissance", type="pdf")


if (
    ("file2" in st.session_state and st.session_state.file2 != file)
    or ("file2" not in st.session_state and file is not None)
    or "agent2" not in st.session_state
):
    st.session_state.file2 = file
    if file is not None:
        file_path = prepare_file(file)
        tmp_docs = DOCUMENTS.copy()
        tmp_docs.append(file_path)
    else:
        tmp_docs = DOCUMENTS.copy()
    st.session_state.agent2 = rag_tool_openai(tmp_docs)

# Display chat messages from history on app rerun
for message in st.session_state.messages2:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = ""
# React to user input
if "agent2" in st.session_state:
    if prompt := st.chat_input("Encore une question ?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages2.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = st.session_state.agent2({"input": prompt})["output"]

# Display assistant response in chat message container
if "agent2" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages2.append({"role": "assistant", "content": response})
