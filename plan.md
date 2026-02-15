# Project Plan: MTG Rules Expert RAG System

## Overview

The **MTG Rules Expert RAG System** is a Retrieval-Augmented Generation
(RAG) application designed to interpret and explain rulings in Magic:
The Gathering using official rules documentation and card data.

The system will: 
- Parse and embed the Comprehensive Rules
- Allow users to input card names and board states
- Retrieve relevant rules sections
- Generate clear, referenced rulings using a local LLM (Ollama)

This tool aims to reduce ambiguity during gameplay and provide fast,
traceable rule explanations.

Owner: Cody Paquette
Status: In-Progress
Model: Ollama (local LLM)
Architecture: Python + Vector Store + Local API + Web Frontend

------------------------------------------------------------------------

## Background

The official Magic: The Gathering Comprehensive Rules exceed 300 pages
and contain highly technical, nested, and conditional language.

In real gameplay scenarios: 
- Players often struggle to locate relevant rules quickly
- Multiple rules interact across sections
- Card text, keywords, state-based actions, and layers complicate rulings

A RAG-based system enables: 
- Semantic retrieval of relevant rule sections
- Citation-backed explanations
- Structured reasoning over card interactions
- Future expansion to judge-level analysis

This project also serves as: 
- A portfolio-grade AI system
- A demonstration of LLM + retrieval architecture
- A customizable foundation for other legal/rule-based RAG systems

------------------------------------------------------------------------

# System Architecture (Phase 1 Target)

1.  Rule Document Ingestion
2.  Text Chunking
3.  Embedding Generation
4.  Vector Storage
5.  Query → Retrieval → Prompt Construction
6.  Ollama Response Generation
7.  Citation Formatting

------------------------------------------------------------------------

## Milestone 1: Basic RAG Prototype (CLI Version)

Target: Functional local command-line system

### Objectives

-   [x] Download and clean Comprehensive Rules text
-   [ ] Implement rule section parser (preserve rule numbers like 603.1,
    704.5a)
-   [x] Chunk rules into structured segments
-   [x] Generate embeddings (local embedding model)
-   [ ] Store embeddings in vector database (FAISS or Chroma)
-   [x] Build retrieval pipeline (top-k similarity search)
-   [x] Construct prompt template for Ollama
-   [x] Generate answer with citations
-   [x] Print response with referenced rule numbers

------------------------------------------------------------------------

## Milestone 2: Card Context Integration

Target: Card-aware reasoning

### Objectives

-   [ ] Integrate card data (Scryfall API or local JSON dump)
-   [ ] Parse card oracle text
-   [ ] Embed oracle text separately
-   [ ] Merge rule + card retrieval results
-   [ ] Implement structured board state input (JSON format)
-   [ ] Improve prompt engineering for interaction reasoning
-   [ ] Add multi-card scenario handling

------------------------------------------------------------------------

## Milestone 3: Web Interface

Target: User-facing interface

### Objectives

-   [ ] Build API wrapper around RAG system (FastAPI)
-   [ ] Design frontend (React or Next.js)
-   [ ] Implement card selection search
-   [ ] Add board state builder UI
-   [ ] Display retrieved rule snippets separately
-   [ ] Add confidence score
-   [ ] Add "Show Sources" expandable panel
-   [ ] Add theme customization (dark MTG aesthetic)

------------------------------------------------------------------------

## Milestone 4: Judge-Level Expansion

Target: Advanced reasoning & structured explanations

### Objectives

-   [ ] Implement structured reasoning outputs
-   [ ] Add rule hierarchy awareness
-   [ ] Add "Why not?" counterfactual explanation mode
-   [ ] Evaluate answer accuracy against known judge rulings
-   [ ] Add logging for failed retrieval cases

------------------------------------------------------------------------

## Risk Assessment & Dependencies

### Risks

-   Poor retrieval quality → Experiment with chunk sizes and embedding
    models
-   Hallucinated rulings → Force citation requirement in prompt template
-   Overly long rule sections → Hierarchical chunking with metadata
    preservation
-   Prompt instability → Versioned prompt templates

### Dependencies

-   Ollama installed locally
-   Embedding model compatibility
-   Official Comprehensive Rules text
-   Card data source (Scryfall or local dataset)

------------------------------------------------------------------------

## Evaluation Metrics

-   Retrieval accuracy
-   Citation correctness
-   Latency (<3 seconds target)
-   Manual validation against known rulings
-   Hallucination frequency

------------------------------------------------------------------------

## Future Expansion

-   Multiplayer board state simulation
-   Layer system visualization
-   Tournament judge mode
-   Voice query support
-   Mobile-friendly interface
-   Deployable cloud version
-   Fine-tuned MTG-specific model

------------------------------------------------------------------------

## Changelog

2026-02-14: Initial structured plan created by Cody Paquette.\
2026-02-14: Started work on prototype for rule interpretation.\
2026-02-14: Built working prototype for rule interpretation.\
2026-02-15: Started work on prototype for card interpretation.\
