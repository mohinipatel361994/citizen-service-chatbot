import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from bhashini_services1 import Bhashini_master
from audio_recorder_streamlit import audio_recorder


# Set up the Streamlit page configuration
st.set_page_config(page_title="Citizen service chatbot", page_icon="🤖")
st.title("Citizen Service Chatbot")
#load_dotenv()

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

# Function to process website content and store the embeddings in session state
def process_website(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(document)
        
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
        embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in document_chunks])
        
        # Storing embeddings and metadata in session_state
        st.session_state.embeddings = embeddings
        st.session_state.document_chunks = document_chunks

        st.success("Website content processed successfully!")
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")

def get_response(user_input):
    if 'embeddings' not in st.session_state or not st.session_state.embeddings:
        st.error("No embeddings found. Please provide a website URL.")
        return ""

    # Create embeddings for the user query
    embeddings_model = OpenAIEmbeddings(api_key=api_key)
    query_embedding = embeddings_model.embed_query(user_input)
    
    # Calculate cosine similarity between query and all document chunks
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(v1, v2):
        return dot(v1, v2) / (norm(v1) * norm(v2))
    
    # Find the most relevant chunk using cosine similarity
    similarities = [
        cosine_similarity(query_embedding, doc_embedding) 
        for doc_embedding in st.session_state.embeddings
    ]
    
    # Get the index of the chunk with highest similarity
    closest_chunk_idx = similarities.index(max(similarities))
    closest_chunk = st.session_state.document_chunks[closest_chunk_idx]
    context = closest_chunk.page_content

    # Use the context directly in the LLM for response generation
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.7)
    
    # Create the messages list using proper message objects
    messages = [
        SystemMessage(content=f"""You are a helpful assistant that provides information based on the given context. 
        Your responses should be accurate and directly related to the context provided.
        If the answer cannot be found in the context, politely state that you don't have enough information to answer the question.
        Always maintain the same language as the user's question."""),
        HumanMessage(content=f"""Context information:
        {context}
        
        User question: {user_input}
        
        Please provide a concise and informative answer based strictly on the above context.""")
                ]
    
    # Get the response from the language model
    response = llm(messages)
    return response.content
    
def get_context_retriever_chain(context):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.7)
    
    prompt_template = """You are a helpful assistant. Answer the following question based on the context provided:

    Context: {context}

    Question: {question}

    Answer:"""

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=None,  # No retriever needed here as we directly pass the context
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "language"],
            ),
        }
    )
    
    return qa_chain


# Sidebar for website URL input
website_url = st.text_input("Paste the URL to process:", placeholder="Enter URL here")

# Initialize session state for chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]

# Process the URL and store data in session state
if website_url and website_url != st.session_state.get('last_url', ''):
    st.session_state.last_url = website_url
    st.session_state.chat_history = []  # Reset chat history for new URL
    with st.spinner("Processing website content..."):
        process_website(website_url)

# Handle audio input and transcription using Bhashini
audio_bytes = audio_recorder("Speak now")
if audio_bytes:
    st.write("Audio recorded. Processing...")
    st.session_state.recorded_audio = audio_bytes
    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=language_code)

    if transcribed_text:
        st.write("Transcribed Audio:", transcribed_text)
        response = get_response(transcribed_text)
        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
        st.session_state.chat_history.append(AIMessage(content=response))
        bhashini_master.speak(response, source_language=language_code)
    else:
        st.write("Error: Audio transcription failed.")

# Handle text input from user
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.spinner("Generating response..."):
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        bhashini_master.speak(response, source_language=language_code)

# Display the chat history
for message in st.session_state.chat_history[-10:]:
    if isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(f"**AIbot:** {message.content}")
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"**User:** {message.content}")
