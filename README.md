# RAG-cover-letter
Powered by Artificial Intelligence, this project generates relevant points indexed on supporting documents in a vector database, retrieves the most relevant experience for each job application

For the first scope, this application will index my CV, previous cover letters, and project notes so the system can retrieve relevant evidence and draft a tailored cover letter for a given job description.”

## Ingest pipeline

Use the LangGraph-powered ingest pipeline to turn source documents into chunked JSONL output and a manifest:

```bash
python src/ingest/ingest.py data/raw -o data/processed
```