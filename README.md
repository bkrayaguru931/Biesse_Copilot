# Biesse AI Support Copilot
<img width="950" height="444" alt="image" src="https://github.com/user-attachments/assets/d0ab4e2a-acc5-4e88-a3d5-6440154bbc44" />

## 1. Statement of Work

AI-powered customer support assistant for Biesse support engineers.

The POC provides:
- real-time speech transcription
- speaker identification
- conversation analysis
- sentiment analysis
- query categorization
- knowledge-grounded RAG retrieval
- AI-generated technical support suggestions

## 2. Client

Biesse

## 3. Project Overview

The system assists a support engineer during a machine-support conversation by converting live audio into a transcript and retrieving relevant technical documentation to generate grounded suggestions.

The AI acts as an assistant to the human support engineer and does not autonomously execute machine operations.

## 4. Current Status

### Completed

- Two-party synthetic customer-support conversation dataset
- Real-time WASAPI loopback audio capture
- AssemblyAI streaming transcription
- Speaker labeling
- Gemini-based conversation analysis
- Gemini Embedding 2 embeddings
- ChromaDB vector store
- RAG retrieval
- Gemini-based grounded suggestion generation
- Browser-based dashboard
- Retrieval evaluation on 10 representative queries
- Feedback driven RAG improvements
- End-to-end local POC demo

### Remaining

- Integration with authorized Biesse technical documentation
- Larger manually labelled RAG evaluation set
- Production-grade telephony/audio integration
- Production deployment infrastructure
- Monitoring and observability
- Production-grade rate limiting/retry handling

## 5. Architecture
<img width="705" height="503" alt="image" src="https://github.com/user-attachments/assets/c3d21933-98a2-435c-9b4e-47191ef10aed" />

Audio
→ WASAPI Loopback
→ Python Backend
→ AssemblyAI Streaming
→ Conversation Processing
→ Gemini Analysis
→ RAG Search
→ ChromaDB
→ Gemini Suggestion Generation
→ SSE
→ Browser Dashboard

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend/orchestration |
| WASAPI Loopback | Local system audio capture |
| AssemblyAI | Streaming transcription + speaker labeling |
| Gemini 3.6 Flash | Conversation analysis + suggestion generation |
| Gemini Embedding 2 | Document/query embeddings |
| ChromaDB | Local vector database |
| PyPDF | PDF text extraction |
| HTML/CSS/JavaScript | Dashboard |
| Server-Sent Events | Backend-to-browser live updates |

## 7. Knowledge Base

The POC currently uses publicly available generic CNC/manufacturing technical references for development and evaluation.

These should not be interpreted as proprietary Biesse documentation.

The knowledge base currently contains:

- 4 technical manuals
- 2,576 pages
- 701 processed chunks
- 1,536-dimensional embeddings

## 8. RAG Pipeline

1. Extract text from PDFs
2. Detect logical sections
3. Chunk technical content
4. Generate embeddings
5. Store vectors in ChromaDB
6. Embed incoming support query
7. Retrieve top-K relevant chunks
8. Provide retrieved context to Gemini
9. Generate grounded support suggestions

## 9. Running the Project

### Create environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then supply valid Gemini and AssemblyAI keys.

### Run

Start the API server:

```bash
python main.py
```

Serve the project directory with VS Code Live Server (or another local static
server) at `http://127.0.0.1:5500`, then open:

```text
http://127.0.0.1:5500/frontend/index.html
```

The first visit displays a sign-up/login screen. Accounts are stored locally in
`data/biesse_auth.db`; passwords are hashed with bcrypt and active sessions
expire after 12 hours. This local account store is appropriate for the POC,
not a production identity provider. If the static server uses a different
origin, set `FRONTEND_ORIGIN` in `.env` to that exact origin.

## 10. Recommendation feedback loop

The Helpful / Not helpful buttons save the reviewed recommendation, its source,
and the conversation context locally. After at least three ratings for the same
document chunk, the RAG pipeline applies a small feedback-based reranking
signal; semantic relevance remains the primary ranking factor.

Export data for manual review or offline evaluation with:

```bash
python rag/export_feedback_dataset.py --helpful-only
```

Do not train directly from raw ratings. Review the exported records, remove
sensitive call content, and use a held-out evaluation set before promoting any
changes to production.
