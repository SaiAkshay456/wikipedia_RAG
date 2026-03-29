import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

def load_documents(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory not found")
    
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents=loader.load()
    for i in range(0,2):
        print(f"Documnet {i+1}")
        print(f" {documents[i].metadata['source']}")
    return documents

def chunk_documents(documents,overlap=0,chunk_size=500):
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks=text_splitter.split_documents(documents)
    if chunks:
        for i in range(0,3):
            print(f"Chunk {i+1}")
            print(f"{chunks[i].page_content}")
    return chunks

def create_embeddings(chunks):
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.getenv("OPENAI_API_KEY"))
    print("started creating embeddings...")
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="db/chroma_db"
    )
    print("embeddings created and stored in vector db")
    return vector_store
def main():
    print("main running")
    #loading files
    #chunking them
    #embeddings of chunks
    #then store in vector db
    #1
    docs=load_documents("docs")
    chunked_docs=chunk_documents(docs)
    create_embeddings(chunked_docs)

if __name__=="__main__":
    main()
