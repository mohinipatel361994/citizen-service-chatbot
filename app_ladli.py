import os
import streamlit as st
import base64
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from bhashini_services1 import Bhashini_master
from audio_recorder_streamlit import audio_recorder
import chromadb
from langchain.chains import RetrievalQA


# Set up the Streamlit page configuration
st.set_page_config(page_title="Citizen service chatbot", page_icon="🤖")
st.title("Citizen Service Chatbot")
# Available languages for selection
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Telugu": "te",
    # Add more languages as required
}

# Step 1: Create a language selection dropdown
selected_language = st.selectbox(
    "Select the language for Question:",
    options=list(languages.keys())
)

# Step 2: Get the corresponding language code based on the user's selection
language_code = languages[selected_language]
st.write(f"Selected language: {selected_language}")

# Fetch API key for OpenAI and other credentials from environment variables
api_key = st.secrets["secret_section"]["openai_api_key"]
bhashini_url = st.secrets["secret_section"]["bhashini_url"]
bhashini_authorization_key = st.secrets["secret_section"]["bhashini_authorization_key"]
bhashini_ulca_api_key = st.secrets["secret_section"]["bhashini_ulca_api_key"]
bhashini_ulca_userid = st.secrets["secret_section"]["bhashini_ulca_userid"]

# Initialize Bhashini master for transcription and translation
bhashini_master = Bhashini_master(
    url=bhashini_url,
    authorization_key=bhashini_authorization_key,
    ulca_api_key=bhashini_ulca_api_key,
    ulca_userid=bhashini_ulca_userid
)

# Create a directory for ChromaDB
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)

def get_vectorstore_from_url(url):
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(document)
        # st.write(f"Document Chunks: {document_chunks[:]}")
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings(api_key=api_key),
            client=client,
            collection_name="website_content"
        )
        return vector_store



def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.7)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    prompt_template = """You are a helpful assistant. Given the following context information from a website:

    {context}

    The user has asked the following question: "{question}"

    Please respond in the same language as the user's question. Provide a concise and informative answer based on the context above.
    Answer:"""

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question","language"],
            ),
        }
    )
    
    return qa_chain


def get_response(user_input):
    if 'vector_store' not in st.session_state:
        st.error("Vector store not found. Please provide a website URL.")
        return ""

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    try:
        response = retriever_chain({"query": user_input, "language": language_code})
        # st.write(f"Raw response from retriever: {response}")
        result = response.get('result', '')
        
        if not result:
            st.error("Received empty response.")
            return ""
        
        return result
    
    except Exception as e:
        st.error(f"Error occurred while retrieving the response: {e}")
        return ""

# Sidebar
website_url = st.text_input("Paste the URL to process:", placeholder="Enter URL here")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
if website_url and website_url != st.session_state.get('last_url', ''):
    st.session_state.last_url = website_url
    st.session_state.chat_history = [] 
    st.session_state.vector_store = None
    # st.cache_data
    with st.spinner("Processing website content..."):
        vector_store = get_vectorstore_from_url(website_url)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.success("Website content processed successfully!")
        else:
            st.error("Failed to process website content. Please check the URL and try again.")
        if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
            st.error("Vector store not found. Please provide a website URL.")

# Handle audio input and transcription using Bhashini
audio_bytes = audio_recorder("Speak now")
if not audio_bytes:
    st.warning("Please record some audio to proceed.")
if audio_bytes:
    st.write("Audio recorded. Processing...")
    st.session_state.recorded_audio = audio_bytes
    file_path = bhashini_master.save_audio_as_wav(audio_bytes, directory="output", file_name="last_recording.wav")
    # Specify language explicitly, assuming input language matches selected language
    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=language_code)

    if transcribed_text:
        st.write("Transcribed Audio:", transcribed_text)
        response = get_response(transcribed_text)  # Get the response in the same language as the input
        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))  # Store user query
        st.session_state.chat_history.append(AIMessage(content=response))  # Store AI response
        # Convert response back to speech (Text-to-Speech)
        bhashini_master.speak(response, source_language=language_code)
    else:
        st.write("Error: Audio transcription failed.")
        transcribed_text = ""

# Handle user input through chat
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.spinner("Generating response..."):
        response = get_response(user_query)  # Get the response in the selected language
        st.session_state.chat_history.append(HumanMessage(content=user_query))  # Store user query
        st.session_state.chat_history.append(AIMessage(content=response))  # Store AI response
        # Convert response to speech if necessary
        bhashini_master.speak(response, source_language=language_code)

# Display the chat history
for message in st.session_state.chat_history[-10:]:
    if isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(f"**AIbot:** {message.content}")
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"**User:** {message.content}")
