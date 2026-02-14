# RAG System with Ollama
This project is uses a Retrieval-Augmented Generation (RAG) system that reads ruling documents (game rules or legal documents) and provides an explanation with references to the supporting documents as to why a situation leads to a certain outcome. 

## What This Does
- Loads rule/law documents
- Breaks document into chunks
- Stores chunks into a vector database
- Retrieves releveant information based on user question
- Uses a local LLM to generate an explanation

## Why I Built It
I wanted to build a system that can explain decisions using actual rules as references.
Instead of guessing, the model retrieves relevant law sections and uses them to justify the answer.

This is the foundation for building more advanced reasoning systems later.

## How It Works
1. Documents are added to the system
2. The documents are converted into embeddings
3. When a user asks a question, similar information sections are retrieved
4. Those information sections are sent to Ollama
5. Ollama generates an explanation using the retrieved rules

## How To Run It
```python rules_rag.py```

## Next Steps
- Add better citation formatting
- Improve rule conflict detection
- Add structured reasoning
- Build a UI
