import streamlit as st
from PIL import Image

st.set_page_config(page_title="Accueil")

st.image(
    Image.open("static/logo_seyna_150_orange.png"),
    width=300,
)

st.title("Accueil")
st.write("Bienvenue sur Seyna AI, notre application de démonstration dédiée à l'IA générative !")
st.write("Cette application est divisée en 2 pages :")
st.write("1. Accueil")
st.write("2. Chatbot 🤖")
st.write("Vous pouvez accéder à ces pages via le menu de gauche.")
st.write("Pour commencer, vous pouvez vous rendre sur la page Chatbot.")
st.write("Bonne visite !")
