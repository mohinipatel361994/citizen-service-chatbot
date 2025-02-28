import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import whisper 
import chromadb
from bhashini_services1 import Bhashini_master
from audio_recorder_streamlit import audio_recorder

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
# Load environment variables
load_dotenv()

# Set up the Streamlit page configuration
st.set_page_config(page_title="Citizen Service Chatbot", page_icon="🤖")
st.title("Citizen Service Chatbot")

# Available languages for selection
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Telugu": "te",
}

# Language selection dropdown
selected_language = st.selectbox(
    "Select the language for Question:",
    options=list(languages.keys())
)

# Get the corresponding language code
language_code = languages[selected_language]
st.write(f"Selected language: {selected_language}")

# Fetch API key for OpenAI
api_key = os.getenv("openai_api_key")

# Bhashini credentials
bhashini_url = os.getenv("bhashini_url")
bhashini_authorization_key = os.getenv("bhashini_authorization_key")
bhashini_ulca_api_key = os.getenv("bhashini_ulca_api_key")
bhashini_ulca_userid = os.getenv("bhashini_ulca_userid")

# Initialize Bhashini master for transcription
bhashini_master = Bhashini_master(
    url=bhashini_url,
    authorization_key=bhashini_authorization_key,
    ulca_api_key=bhashini_ulca_api_key,
    ulca_userid=bhashini_ulca_userid
)
whisper_model = whisper.load_model("base")
# Create a directory for ChromaDB
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

def get_chroma_vectorstore():
    """Load the vector store from the persistent ChromaDB directory."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_function = OpenAIEmbeddings(api_key=api_key) 
    vector_store = Chroma(client=client, collection_name="website_content", embedding_function=embedding_function)
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0)
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    prompt_template = """You are a helpful assistant with expert knowledge. Below is the relevant information that you know about the topic:

    {context}

    The user has asked the following question:

    "{question}"

    Please provide a detailed, relevant answer based on your knowledge. If your knowledge doesn't fully address the user's question, kindly suggest where they can find more information or ask for clarification.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question","language"],
            ),
        }
    )
    
    return qa_chain

def get_response(user_input):
    vector_store = get_chroma_vectorstore()
    if not vector_store:
        st.error("Vector store not found. Please provide a website URL.")
        return "Sorry, I couldn't retrieve the information."

    retriever_chain = get_context_retriever_chain(vector_store)

    try:
        response = retriever_chain.invoke({"query": user_input, "language": language_code})
        result = response.get('result', "Sorry, I couldn't find specific details on that topic. Could you please rephrase your question or ask something else?")
        # Extract relevant source URLs
        source_urls = []
        seen_urls = set()
        for doc in response.get("source_documents", []):
            url = doc.metadata.get("source")  # Get the source URL from the document metadata
            if url and url not in source_urls:
                seen_urls.add(url)
                source_urls.append(url)

        # Prepare the final response message
        final_response = final_response = f"Based on the information I have, here's the answer to your question: \n{result}"

        # Add references section if there are any source URLs
        if source_urls:
            final_response += "\n\nReferences:\n" + "\n".join(f"- [Source]({url})" for url in source_urls)
        else:
            final_response += "\n\nNo direct source links were available for this query."

        return final_response    
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."
    
def detect_language_with_whisper(audio_bytes):
    
    # Load the audio file using Whisper
    audio = whisper.load_audio(audio_bytes)
    audio = whisper.pad_or_trim(audio)
    
    # Make a prediction to detect the language
    _, probs = whisper_model.detect_language(audio)
    
    # Get the language with the highest probability
    detected_language = max(probs, key=probs.get)
    
    return detected_language

# Handle audio input and transcription using Bhashini
audio_bytes = audio_recorder("Speak now")
if not audio_bytes:
    st.warning("Please record some audio to proceed.")
else:
    st.write("Audio recorded. Processing...")
    st.session_state.recorded_audio = audio_bytes
    file_path = bhashini_master.save_audio_as_wav(audio_bytes, directory="output", file_name="last_recording.wav")
    detected_language = detect_language_with_whisper(file_path)
    st.write(f"Detected Language: {detected_language}")
    # Specify language explicitly, assuming input language matches selected language
    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=language_code)

    if transcribed_text:
        st.write(f"Transcribed Audio: {transcribed_text}")
        response = get_response(transcribed_text)  # Get the response in the same language as the input
        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))  # Store user query
        st.session_state.chat_history.append(AIMessage(content=response))  # Store AI response
        # Convert response back to speech (Text-to-Speech)
        bhashini_master.speak(response, source_language=language_code)
    else:
        st.write("Error: Audio transcription failed.")
        transcribed_text = ""

# Chat input box for manual text input
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.spinner("Generating response..."):
        response = get_response(user_query)  # Get the response in the selected language
        st.session_state.chat_history.append(HumanMessage(content=user_query))  # Store user query
        st.session_state.chat_history.append(AIMessage(content=response))  # Store AI response
        # Convert response to speech if necessary
        bhashini_master.speak(response, source_language=language_code)

# Display the chat history
if "chat_history" in st.session_state:
    for message in st.session_state.chat_history[-10:]:
        if isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.markdown(f"**AIbot:** {message.content}")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"**User:** {message.content}")
