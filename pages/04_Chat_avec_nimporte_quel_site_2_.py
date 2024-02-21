import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatGPT
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests

# Load the OpenAI API key from the environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def fetch_website_content(url):
    """
    Fetches content from the specified URL and returns plain text.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    return text

def generate_questions(text, openai_api_key, num_questions=3):
    """
    Generates questions based on the provided text using OpenAI's API.
    """
    openai = ChatGPT(api_key=openai_api_key)
    prompt = f"Please generate {num_questions} insightful questions based on the following content:\n{text[:2000]}"
    response = openai.generate(prompt=prompt, max_tokens=100, temperature=0.7, n=num_questions)
    questions = [resp.strip() for resp in response.split('\n') if resp.strip()]
    return questions[:num_questions]

def initialize_session():
    """
    Initializes session state variables.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []

# Streamlit UI
st.set_page_config(page_title="Chat with Web Content")
st.title("Chat with Web Content")

initialize_session()

website_url = st.sidebar.text_input("Enter website URL:", key="url")

if website_url:
    if 'last_url' not in st.session_state or st.session_state.last_url != website_url:
        with st.spinner('Fetching website content...'):
            website_content = fetch_website_content(website_url)
            st.session_state.generated_questions = generate_questions(website_content, OPENAI_API_KEY)
            st.session_state.last_url = website_url
            st.session_state.chat_history.clear()

    st.subheader("Generated Questions")
    for question in st.session_state.generated_questions:
        if st.button(question):
            # Here, add logic to handle the question, e.g., display a predefined answer or use it as input for further processing.
            st.session_state.chat_history.append({'role': 'user', 'text': question})
            # Simulate an AI response for demonstration purposes
            st.session_state.chat_history.append({'role': 'AI', 'text': "This is a simulated response based on the question."})

    st.subheader("Conversation")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("You"):
                st.write(message['text'])
        elif message['role'] == 'AI':
            with st.chat_message("AI"):
                st.write(message['text'])

else:
    st.write("Please enter a URL to start.")
