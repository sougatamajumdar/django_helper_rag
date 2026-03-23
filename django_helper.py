import os
import glob
from dotenv import load_dotenv
import gradio as gr


# The Modern Stack
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The Bridge (Legacy) Stack
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Standard Utilities
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


MODEL = "gemini-3-flash-preview"  
db_name = "django_helper_db"


load_dotenv(override=True)


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


folders = glob.glob("rag_data/*", recursive=True)
print("Folders found:", folders)

def add_metadata(doc, doc_type):
    doc.metadata["type"] = doc_type
    return doc

text_loader_kwargs = {"encoding": "utf-8"}

documents = []

md_loader = DirectoryLoader(
    "rag_data",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
documents.extend(md_loader.load())


py_loader = DirectoryLoader(
    "rag_data",
    glob="**/*.py",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
documents.extend(py_loader.load())


json_loader = DirectoryLoader(
    "rag_data",
    glob="**/*.json",
    loader_cls=TextLoader,   # safer than JSONLoader for mixed format
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
documents.extend(json_loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Sample chunk metadata:", chunks[0].metadata)
print("Total chunks:", len(chunks))


for doc in documents:
    parts = doc.metadata["source"].split(os.sep)
    doc.metadata["domain"] = parts[1] if len(parts) > 1 else "unknown"



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")



collection = vectorstore._collection
sample_embeddings = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embeddings)
print(f"the vectors have {dimensions:,} dimesion")


retriever = vectorstore.as_retriever( search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


response = conversation_chain({"question": "what is the purpose of the file named 'views.py' in the 'django'?"})
print(response["answer"])


custom_template = """
You are a Django Expert Assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and focus on code structure.

Context: {context}
Chat History: {chat_history}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=custom_template, 
    input_variables=["context", "chat_history", "question"]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

def predict(message, history):
    # 'history' from Gradio is a list of lists [[user, bot], [user, bot]]
    # Our 'memory' object already handles this internally, so we just call the chain
    response = qa_chain({"question": message})
    return response["answer"]

# 4. LAUNCH GRADIO
demo = gr.ChatInterface(
    fn=predict,
    title="Django Code Helper",
    description="Ask me anything about your RAG data files!",
    examples=["What is in views.py?", "Explain the project structure."]
)

if __name__ == "__main__":
    demo.launch()


# 
