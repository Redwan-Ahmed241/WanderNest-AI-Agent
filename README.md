# WanderNest RAG - Quick Start

1) .env setup (required)
- GOOGLE_API_KEY=your_google_api_key
- DOCUMENTS_FOLDER=path_to_your_documents (e.g. e:/WanderNest-Agent/documents)
- CHROMA_PERSIST_DIR=path_to_chroma_db (e.g. e:/WanderNest-Agent/chroma_db)

2) Add documents
- Put all your PDF/TXT files into the folder set by DOCUMENTS_FOLDER.

3) Start the server
```bash
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
python main.py
```

4) Use the /query API
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```
