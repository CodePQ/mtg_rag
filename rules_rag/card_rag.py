from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load in rules file
print("Reading files and generating documents...")
csv_path = Path(__file__).resolve().parent/"data"/"cleaned_cards.csv"
loader = CSVLoader(file_path=str(csv_path), encoding='utf-8')
documents = loader.load()
print("Documents generated.")

# Create embeddings for documents and store them in a vector store
print("Embedding documents...")
vectorstore = SKLearnVectorStore.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
print("Documents embeded.")

retriever = vectorstore.as_retriever(k=5)

prompt = PromptTemplate.from_template(
    """
    You are an MTG card explainer.

    Use ONLY the information in the provided CONTEXT.
    If the context does not contain required information (for example, a keyword definition), clearly state what is missing and do not guess.

    Your goal is to explain the card in clear, beginner-friendly language while remaining technically accurate.

    When you answer, follow this exact structure:

    A) Plain-English Summary  
    (2-6 sentences explaining what the card does overall.)

    B) Step-by-Step Breakdown  
    - Break down how the card works in order.
    - If the card has multiple abilities, explain each separately.

    C) Keywords  
    - List each keyword found on the card (if any).
    - For each keyword, provide its definition using ONLY the provided context.
    - If a keyword definition is not present in the context, state: "Definition for [keyword] not found in provided context."

    D) Common Misunderstandings (Optional)  
    - Include up to 3 bullet points.
    - Only include these if supported by the provided context.

    E) Sources Used  
    - List the names or identifiers of the context sections you used.

    Do NOT:
    - Invent rules text.
    - Use outside knowledge.
    - Assume interactions with other cards unless they are included in the context.

    If the question asks about an interaction with another card that is not included in the context, respond:
    "Additional card information required."

    ----------------------------------------

    CONTEXT:
    {context}

    ----------------------------------------

    USER QUESTION:
    {question}

    ----------------------------------------

    DOCUMENTATION:
    {documents}

    ----------------------------------------

    ANSWER:

    """
)

# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0.8,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke(
            {"question": question, "documents": doc_texts})
        return answer


# Initialize the RAG application
print("Initializing RAG...")
rag_application = RAGApplication(retriever, rag_chain)
print("RAG initialized.\n")
running = True
while running:
    # Example usage
    question = input("Enter quesiton: ")
    if question == 'quit':
        running = False
    else:
        answer = rag_application.run(question)
        print("Answer:", answer)
    print(150*"=")
    print("\n")
