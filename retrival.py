import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
embeddings=OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.getenv("OPENAI_API_KEY"))
point_chroma=Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

retriver=point_chroma.as_retriever(search_kwargs={"k":3})

relevant_docs=retriver.invoke("In which year netflix established or started?")
for i in range(0,3):
    print(f"Document {i+1}")
    print(f"{relevant_docs[i].page_content}")
