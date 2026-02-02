# Setup Guide for Agentic AI RAG Chatbot

Welcome! This guide will help you set up and run the Agentic AI RAG Chatbot on your own computer. This project is designed to be easy to run, even if you are new to Python or AI.

## 1. Prerequisites (What you need installed)

Before you begin, make sure you have **Python** installed.
- **Check if you have Python**: Open your terminal (Command Prompt or PowerShell) and type `python --version`.
- **If not installed**: Download and install it from [python.org](https://www.python.org/downloads/). Make sure to check the box **"Add Python to PATH"** during installation.

## 2. Get the Code

If you are reading this on GitHub, you likely already cloned the repository. If not:
1.  Open your terminal.
2.  Run: `git clone <repository-url>`
3.  Go into the project folder: `cd rag-agentic-ai`

## 3. Install Dependencies

We need to install the libraries that make this bot work (like LangChain, Streamlit, etc.).
1.  In your terminal, inside the `rag-agentic-ai` folder, run:
    ```bash
    pip install -r requirements.txt
    ```
2.  Wait for it to finish installing all the packages.

## 4. Set Up Your API Keys (Important!)

This chatbot uses Google Gemini (for intelligence) and Pinecone (for memory). You need secure keys for them.

**Step A: Get a Google Gemini API Key**
1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Click "Create API Key".
3.  Copy the key.

**Step B: Get a Pinecone API Key**
1.  Go to [Pinecone.io](https://www.pinecone.io/) and sign up (it's free).
2.  Create an "Index":
    - **Name**: `agentic-ai-rag-768`
    - **Dimensions**: `768`
    - **Metric**: `cosine`
    - **Cloud**: AWS
    - **Region**: `us-east-1` (or whatever is free/available, just remember it).
3.  Go to "API Keys" section and copy your API Key.

**Step C: Create the .env File**
1.  In the `rag-agentic-ai` folder, create a new text file named `.env`. (Make sure it's just `.env`, not `.env.txt`).
2.  Paste the following inside, replacing the text with your actual keys:

    ```env
    GOOGLE_API_KEY=paste_your_google_key_here
    PINECONE_API_KEY=paste_your_pinecone_key_here
    PINECONE_ENV=us-east-1
    PINECONE_INDEX_NAME=agentic-ai-rag-768
    ```
    *(Note: If you chose a different region in Pinecone, update PINECONE_ENV).*

## 5. Add Your Data

1.  Find the file `Ebook-Agentic-AI.pdf`.
2.  Place it inside the `data` folder: `rag-agentic-ai/data/Ebook-Agentic-AI.pdf`.

## 6. Run the Project

Now we are ready to launch!

**Step A: Ingest Data (Do this once)**
Run this command to process the PDF and save it to the database:
```bash
python ingest.py
```
*You should see a message saying "Pinecone ingestion complete" or "FAISS index saved".*

**Step B: Start the Chatbot**
Run this command to start the user interface:
```bash
streamlit run streamlit_app.py
```

This will open a new tab in your web browser where you can chat with the AI!

## Troubleshooting

- **"Model not found" error?**
  Open `llm.py` and verify that the model name matches what is available in your Google AI Studio account (e.g., `gemini-2.5-flash` or `gemini-1.5-flash`).

- **Pinecone dimension error?**
  Make sure your Pinecone Index has **768 dimensions**. If you created one with 1024 or 1536, delete it and create a new one with 768.

Enjoy this chatbot and tell me if any improvements needed!
