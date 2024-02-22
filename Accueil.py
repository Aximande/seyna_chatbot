import streamlit as st
from PIL import Image

st.set_page_config(page_title="Accueil")

st.image(
    Image.open("static/logo_seyna_150_orange.png"),
    width=300,
)

st.title("Accueil")
st.write("""
Bienvenue sur Seyna AI, notre application de démonstration dédiée à l'IA générative pour les assurances et bien plus.
Cette application vous permet d'explorer une gamme de fonctionnalités avancées d'IA, conçues pour améliorer l'efficacité et l'interaction dans divers contextes professionnels. Voici un aperçu des pages disponibles :
""")

# Liste des pages avec descriptions
pages = {
    "Chatbot Intelligent": "Dialogue avec un chatbot alimenté par IA pour répondre à vos questions courantes.",
    "Assistant FAQ Assurantiel": "Utilise RAG pour fournir des réponses précises à partir d'une base de connaissances spécialisées en assurances.",
    "Comparaison de Tableaux CSV": "Comparez et analysez des données à partir de fichiers CSV de manière intuitive.",
    "Navigation par Chat": "Interagissez avec n'importe quel site Web via un chatbot pour une navigation simplifiée.",
    "Extracteur de Bulletins d'Adhésion Santé": "Extrait les informations clés des bulletins d'adhésion à des assurances santé.",
    "Extracteur de Données de Bail": "Récupérez et structurez les informations essentielles des contrats de bail."
}

# Affichage des pages et leurs descriptions
for page, description in pages.items():
    st.write(f"- **{page}**: {description}")

st.write("""
Accédez à ces fonctionnalités via le menu de navigation à gauche. Commencez par explorer notre Chatbot Intelligent pour une introduction interactive à nos services.
Bonne visite et découverte !
""")
