import os
import asyncio
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Asynchronous function to load multiple documents
async def load_documents_async(file_paths):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, load_document, file_path) for file_path in file_paths]
    documents = await asyncio.gather(*tasks)
    return [doc for sublist in documents for doc in sublist]

def load_document(file_path):
    # Use PyPDFLoader from langchain to load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Use st.cache_data to cache the vector store
@st.cache_data(persist="disk", show_spinner=True)
def setup_vectorstore(_documents):  # Renamed 'documents' to '_documents' to avoid hashing
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(_documents)  # Now using _documents
    
    # Create a FAISS vectorstore from the document chunks and return it
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore


def create_chain(vectorstore):
    # Use ChatGroq model for the LLM with specific settings
    llm = ChatGroq(
        model="llama-3.2-3b-preview",  # Model you requested
        temperature=0,            # Set temperature to 0
        top_p=0.9                 # Set top-p to 0.9
    )
    
    retriever = vectorstore.as_retriever()
    
    # Memory to maintain the conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create a conversational chain with LLM and vectorstore retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit app settings
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with PDF - LLAMA 3.2")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_files = st.file_uploader(label="Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # Load documents asynchronously
    documents = asyncio.run(load_documents_async(file_paths))
    
    # Clear cache if new documents are uploaded
    if "uploaded_files" not in st.session_state or st.session_state.uploaded_files != uploaded_files:
        st.cache_data.clear()  # Clear cached vector store
        st.session_state.uploaded_files = uploaded_files  # Track the new files
    
    # Setup vector store and cache it using Streamlit's st.cache_data
    vectorstore = setup_vectorstore(documents)
    
    # Reset the conversation chain when new files are uploaded
    st.session_state.conversation_chain = create_chain(vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("Ask Llama...")

if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from the LLM
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
