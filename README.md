# Biesse AI Support Copilot

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
- End-to-end local POC demo

### Remaining

- Integration with authorized Biesse technical documentation
- Larger manually labelled RAG evaluation set
- Production-grade telephony/audio integration
- Authentication and authorization
- Production deployment infrastructure
- Monitoring and observability
- Production-grade rate limiting/retry handling

## 5. Architecture

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