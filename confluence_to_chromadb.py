import os
import time
from configparser import ConfigParser
from langchain_community.document_loaders import ConfluenceLoader
import chromadb
from chromadb import Client
from chromadb.config import Settings
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OllamaEmbeddings
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



# def split_and_embed(data, embedder):
#     """Split the data into chunks and embed it using Ollama."""
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_text(data)
#     embeddings = [embedder.embed(chunk) for chunk in chunks]
#     return chunks, embeddings

# def store_in_chromadb(chroma_client, collection_name, chunks, embeddings):
#     """Store data chunks and embeddings in ChromaDB."""
#     collection = chroma_client.get_or_create_collection(name=collection_name)
#     for chunk, embedding in zip(chunks, embeddings):
#         collection.add(documents=[chunk], embeddings=[embedding])

def process_document() -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Confluence configuration
    #read config
    config = ConfigParser()
    config.read("config.ini")
    username= config["confluence_credentials"]["username"]
    passkey= config["confluence_credentials"]["passkey"]
    confluence_url= config["confluence_credentials"]["url"]
    spacekey= config["confluence_credentials"]["spacekey"]

    # Store uploaded file as a temp file
    loader = ConfluenceLoader(
    url=confluence_url,
    username=username,
    api_key=passkey,
    space_key=spacekey,  # Replace with the Confluence space key
    include_attachments=False  # Set True if you want to include attachments
    )

    # Load Documents
    documents = loader.load()

    # Process Documents
    for doc in documents:
        print(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(documents)

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print("Data added to the vector store!")
    

def main():
    # Confluence configuration
    #read config
    # config = ConfigParser()
    # config.read("config.ini")
    # username= config["confluence_credentials"]["username"]
    # passkey= config["confluence_credentials"]["passkey"]
    # confluence_url= config["confluence_credentials"]["url"]
    # spacekey= config["confluence_credentials"]["spacekey"]

    # print("Fetching data from Confluence...")
    # loader = ConfluenceLoader(
    # url=confluence_url,
    # username=username,
    # api_key=passkey,
    # space_key=spacekey,  # Replace with the Confluence space key
    # include_attachments=False  # Set True if you want to include attachments
    # )

    # # Load Documents
    # documents = loader.load()

    # # Process Documents
    # for doc in documents:
    #     print(doc.page_content)

    # ChromaDB configuration
    # CHROMADB_PATH = "./chroma_db"    

    # # Initialize ChromaDB client
    # chroma_client = Client(
    #     Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=CHROMADB_PATH
    #     )
    # )

    # Initialize Ollama embeddings
    #embedder = OllamaEmbeddings(model="local_llm_model")

    all_splits = process_document()
    add_to_vector_collection(all_splits, "confluence_file")
    

    # print("Fetching data from Confluence...")
    # data = fetch_confluence_data(confluence, SPACE_KEY, PAGE_TITLE)
    # if data:
    #     print("Splitting and embedding data...")
    #     chunks, embeddings = split_and_embed(data, embedder)

    #     print("Storing data in ChromaDB...")
    #     store_in_chromadb(chroma_client, "confluence_data", chunks, embeddings)

if __name__ == "__main__":
    main()