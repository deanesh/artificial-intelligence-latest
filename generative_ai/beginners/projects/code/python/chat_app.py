import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Set Streamlit page config
st.set_page_config(page_title="Local Chat with Mistral", layout="wide")

st.title("🧠 Local AI Chatbot using Ollama + LangChain")

# Initialize chat memory and model
@st.cache_resource
def load_conversation():
    llm = Ollama(model="mistral")  # Make sure Mistral is pulled with `ollama pull mistral`
    memory = ConversationBufferMemory()
    convo = ConversationChain(llm=llm, memory=memory)
    return convo

conversation = load_conversation()

# Chat interface
user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = conversation.run(user_input)
        st.markdown(response)
