# RAG Chatbot Streamlit App

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, FAISS, and Hugging Face models. This project allows you to input a website URL or upload a document (PDF, TXT, DOCX), index its content, and ask natural language questions with quick, accurate answers.

## 🚀 Features

- **Website & File Inputs**: Paste any public URL or upload a PDF/TXT/DOCX file.
- **Progress Feedback**: Real-time green progress bar and status messages during processing.
- **Text Extraction**: HTML parsing with BeautifulSoup; PDF, DOCX, and TXT support.
- **Chunking & Embedding**: Splits text into chunks and creates embeddings with SentenceTransformer.
- **Vector Search**: Builds a FAISS index for fast similarity search.
- **LLM Inference**: Uses free Hugging Face models (e.g., Falcon, T5) to generate answers.
- **Prompt Refinement**: Ensures the model only outputs concise answers without echoing headers.
- **Context Display**: Collapsible context snippets so you can see where answers come from.
- **Index Persistence**: Saves and reloads FAISS index and chunks to disk for instant startup.
- **Deployment Ready**: Easily deploy to Streamlit Cloud or Hugging Face Spaces.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-chatbot-streamlit.git
   cd rag-chatbot-streamlit
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

- Use the sidebar to select **Website URL** or **Upload File**.
- Click **Process & Index** and wait for the ✅ completion message.
- Ask your question in the chat box and press Enter.

## ☁️ Deployment

### Streamlit Cloud

1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and deploy a new app from your repository.

### Hugging Face Spaces

1. Push your code to GitHub.
2. Create a new Space on Hugging Face, select **Streamlit** as the SDK, and link your repo.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

