import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="URL Upload and Vector Store", page_icon="🌐")
st.title("URL Upload and Vector Store")

# Fetch API key for OpenAI and other credentials from environment variables
api_key = os.getenv("openai_api_key")

# Create a directory for ChromaDB
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Define the metadata file and log file
metadata_file = os.path.join(PERSIST_DIR, "vector_store_metadata.txt")
log_file = os.path.join(PERSIST_DIR, "vector_store_creation_log.txt")

def log_vector_store_creation(url, timestamp, status):
    """Log the vector store creation event in a log file.
       If the URL is already logged, update its status and timestamp."""
    log_entry = f"{timestamp} | URL: {url} | Status: {status}\n"
    
    # Read existing log file and check if the URL already exists
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            log_content = log.readlines()
        
        # Update the log if the URL is already in the file
        updated = False
        with open(log_file, "w") as log:
            for line in log_content:
                if url in line:
                    log.write(log_entry)  # Replace the previous log entry with the new one
                    updated = True
                else:
                    log.write(line)
            
            # If the URL was not found, append the new entry
            if not updated:
                log.write(log_entry)
    else:
        # If the log file doesn't exist, create it and write the log entry
        with open(log_file, "w") as log:
            log.write(log_entry)

def get_vectorstore_from_url(url): 
    """Process the website content and create a vector store."""
    try:
        # Load the website content
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(document)

        client = chromadb.PersistentClient(path=PERSIST_DIR)
        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings(api_key=api_key),
            client=client,
            collection_name="website_content"
        )
        if vector_store is None:
            st.error("We couldn't process the website. Please check the URL format and ensure it is publicly accessible.")
            log_vector_store_creation(url, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Failed")
            return None

        # Save metadata with timestamp of vector store creation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_file, "w") as file:
            file.write(f"URL: {url}\nCreated At: {timestamp}\n")

        # Log the successful creation or update the log
        log_vector_store_creation(url, timestamp, "Success")

        return vector_store, timestamp

    except Exception as e:
        st.error(f"Error processing the website: {e}")
        log_vector_store_creation(url, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Failed")
        return None, None

def get_chroma_vectorstore():
    """Load the vector store from the persistent ChromaDB directory."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_function = OpenAIEmbeddings(api_key=api_key) 
    vector_store = Chroma(client=client, collection_name="website_content", embedding_function=embedding_function)
    return vector_store

# Sidebar for URL input
website_url = st.text_input("Paste the URL to process:", placeholder="Enter URL here")

if website_url:
    with st.spinner("Processing website content..."):
        vector_store, timestamp = get_vectorstore_from_url(website_url)
        if vector_store:
            st.success(f"Website content processed successfully at {timestamp}!")
        else:
            st.error("Failed to process website content. Please check the URL and try again.")

# Load the vector store directly from the persistent ChromaDB directory
if st.button("Load Vector Store from Database"):
    with st.spinner("Loading vector store..."):
        vector_store = get_chroma_vectorstore()
        if vector_store:
            st.success("Vector store loaded successfully!")
        else:
            st.error("Failed to load vector store from the database.")

# Display log of vector store creation events
if os.path.exists(log_file):
    st.subheader("Vector Store Creation Log")
    with open(log_file, "r") as file:
        log_content = file.read()
        st.text_area("Log History", log_content, height=300)
