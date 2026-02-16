import ingest_testing
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load in rules file
print("Reading files...")
csv_file_path = "rules_rag/data/cleaned_cards.csv"
loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')
documents = loader.load()

with open('rules_rag/data/MagicCompRules_20260116.txt', 'r', encoding='utf-8') as file:
    rules = file.read()
rules_list = rules.split('\n\n')

for rule in rules_list:
    doc = Document(page_content=rule)
    documents.append(doc)

print("Files loaded.")

# Create embeddings for documents and store them in a vector store
print("Embedding chunks...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Chunks embeded.")

vectorstore = ingest_testing.get_or_build_vectorstore(documents, embeddings)

retriever = vectorstore.as_retriever(k=10)

prompt = PromptTemplate.from_template(
    """
    You are an expert-level Magic: The Gathering rules judge.

    You answer questions strictly using the provided context from the official Comprehensive Rules document.
    You do not rely on outside knowledge.
    If the answer is not fully supported by the provided rules context, you must say:
    "I cannot determine the ruling from the provided rules context."

    Your task:
    Given a game scenario or rules question, determine the correct ruling using only the provided rules excerpts.

    Instructions:

    1. Carefully analyze the scenario.
    2. Identify which rules apply.
    3. Quote the relevant rule numbers and text.
    4. Explain step-by-step how those rules apply to the situation.
    5. Provide a clear final ruling.

    If the scenario is ambiguous or missing required details:
    - Explicitly state what information is missing.
    - Explain how the ruling could change depending on that missing information.

    Formatting Requirements:

    Answer in this structure:

    ---
    **Relevant Rules**
    - Rule [number]: "Quoted rule text"
    - Rule [number]: "Quoted rule text"

    **Analysis**
    Step-by-step explanation applying the rules to the scenario.

    **Final Ruling**
    A clear, concise statement of what happens in the game.

    ---

    Context from the Comprehensive Rules:
    {context}

    Question:
    {question}
    """
)

# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0.8,
)

# Create a chain combining the prompt template and LLM
rag_chain = {"context": retriever, "question": RunnablePassthrough()
             } | prompt | llm | StrOutputParser()

running = True
while running:
    question = input("Enter quesiton: ")
    if question == 'quit':
        running = False
    else:
        answer = rag_chain.invoke(question)
        print("Answer:", answer)
    print(150*"=")
