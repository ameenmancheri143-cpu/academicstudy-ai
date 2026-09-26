# 🎓 Academic Study AI Assistant

A privacy-first, dual-mode local AI assistant designed to help students analyze textbook PDFs and answer general academic queries. Built entirely with open-source tools, this application runs entirely locally on your machine, ensuring zero data leakage to external cloud APIs.

## 🚀 Features

* **Dual-Mode Intelligence:** Switch seamlessly between "General Knowledge" mode and "Study my PDF" mode.
* **Retrieval-Augmented Generation (RAG):** Upload a textbook or lecture notes (PDF), and the AI will analyze the text, extract context via semantic search, and provide accurate answers grounded *only* in your uploaded material.
* **Local Processing:** Powered by Ollama running a localized instance of Llama 3.2.
* **Interactive UI:** Built with Streamlit, featuring a clean chat interface, conversation history, and adjustable AI temperature settings.

## 🛠️ Technology Stack

* **Frontend:** Streamlit
* **LLM Engine:** Ollama (Model: Llama 3.2)
* **Embeddings & Vector Store:** HuggingFace (`all-MiniLM-L6-v2`), FAISS
* **Text Processing:** LangChain, PyPDF2
* **Language:** Python 3

## 💻 Installation & Setup

To run this project locally, you will need Python and Ollama installed on your machine.

### 1. Install Ollama & the LLM
1. Download and install [Ollama](https://ollama.com/).
2. Open your terminal and pull the Llama 3.2 model by running:
   ```bash
   ollama run llama3.2
