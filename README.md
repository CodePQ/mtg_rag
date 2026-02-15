# MTG Rules Expert RAG System

A Retrieval-Augmented Generation (RAG) system designed specifically to
interpret and explain rulings in **Magic: The Gathering** using the
official Comprehensive Rules and card data.

This system retrieves relevant rule sections, constructs a structured
prompt, and generates a citation-backed explanation using a local LLM
powered by **Ollama**.

------------------------------------------------------------------------

## Project Purpose

Magic: The Gathering's Comprehensive Rules exceed 300 pages and contain
highly technical, hierarchical, and conditional language.

This project aims to:

-   Retrieve the exact rule sections relevant to a gameplay situation
-   Generate structured explanations grounded in official rules
-   Provide clear rule citations
-   Reduce hallucination through retrieval grounding
-   Serve as a scalable foundation for judge-level reasoning tools

This is an MTG-focused system first, designed for future expansion.

------------------------------------------------------------------------

## System Architecture

Phase 1 Architecture:

1.  Rule Document Ingestion
2.  Text Chunking (preserving rule numbers like 603.1, 704.5a)
3.  Embedding Generation (local embedding model)
4.  Vector Storage (FAISS or Chroma)
5.  Query → Retrieval → Prompt Construction
6.  Ollama Response Generation
7.  Citation Formatting

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   Ollama (local LLM runtime)
-   FAISS or Chroma (vector database)
-   Local embedding model
-   Structured rule parsing

------------------------------------------------------------------------

## Current Milestones

### Milestone 1: Basic RAG Prototype (CLI)

-   Download and clean Comprehensive Rules text
-   Parse rule sections with numbering preserved
-   Chunk rules into structured segments
-   Generate embeddings
-   Store embeddings in vector database
-   Implement top-k retrieval
-   Construct structured prompt template
-   Generate explanation with citations

------------------------------------------------------------------------

### Milestone 2: Card Context Integration

-   Integrate card data (Scryfall or local JSON)
-   Parse oracle text
-   Embed card data separately
-   Merge rule + card retrieval
-   Support structured board state input (JSON)
-   Handle multi-card interactions

------------------------------------------------------------------------

### Milestone 3: Web Interface

-   FastAPI backend wrapper
-   Custom frontend (React or Next.js)
-   Card search + board state builder
-   Source panel for retrieved rule snippets
-   Confidence scoring
-   Dark MTG-themed UI

------------------------------------------------------------------------

### Milestone 4: Judge-Level Expansion

-   Structured reasoning output
-   Rule hierarchy awareness
-   Counterfactual explanations ("Why not?" mode)
-   Accuracy validation against known rulings
-   Logging and retrieval diagnostics

------------------------------------------------------------------------

## Evaluation Metrics

-   Retrieval accuracy
-   Citation correctness
-   Latency target: < 3 seconds (local)
-   Manual validation against official rulings
-   Hallucination frequency

------------------------------------------------------------------------

## Roadmap

-   Improve citation formatting
-   Implement rule conflict detection
-   Add structured reasoning layers
-   Expand card interaction depth
-   Build scalable web deployment
-   Add judge-mode advanced reasoning

------------------------------------------------------------------------

## Author

Cody Paquette\
Data Science \| RAG Systems \| Rule-Based AI Architectures
