# 📚 Django DRF RAG Helper

An AI-powered study assistant for Django and Django REST Framework (DRF) built using a Retrieval-Augmented Generation (RAG) pipeline. It helps with concept understanding, debugging, and code guidance using your own curated knowledge base.

---

##  Features

- Context-aware Q&A from Django/DRF docs, code, and error cases  
- Code explanation and debugging assistance  
- Conversational chat with memory  
- Custom knowledge base generation using local LLM (Ollama)  
- Simple Gradio chat interface  

---

## How It Works

1. Load `.md`, `.py`, and `.json` files from `rag_data/`  
2. Split into chunks using LangChain  
3. Convert chunks into embeddings  
4. Store in Chroma vector database  
5. Retrieve relevant context for each query  
6. Generate grounded answers using an LLM  

---

## Project Structure

```
rag_data/        # Generated knowledge base
app.py           # Main RAG + Gradio app
generate_data.py # Script to create RAG data using Ollama
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create `.env`:

```
OPENAI_API_KEY=your_api_key
```

---

## Generate RAG Data

Make sure Ollama is running:

```bash
ollama run qwen2.5:3b-instruct
```

Then:

```bash
python generate_data.py
```

---

## Run the App

```bash
python app.py
```

Open in browser:
```
http://127.0.0.1:7860
```

---

## Example Queries

- What is the purpose of views.py?  
- Explain ModelSerializer  
- Common DRF validation errors  
- Best practices for Django project structure  

---

## Tech Stack

- LangChain  
- Chroma (Vector DB)  
- HuggingFace Embeddings  
- Gradio  
- Ollama (for data generation)  
- Gemini (for response generation)  

---

## Notes

- Data is generated and stored locally  
- Easily extendable with more documents  
- Supports hybrid LLM setup (API + local models)
