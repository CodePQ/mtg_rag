from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load in rules file
print("Reading files...")
rules_db = []
with open('data/mtg_rules_20260116.txt', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        clean_line = line.strip()
        if clean_line != '':
            rules_db.append(line.strip())
    print(f'Loaded {len(rules_db)} entries')
print("Files loaded.")

# Initialize a text splitter with specified chunk size and overlap
print("Splitting files into chunks...")
"""text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=150
)"""
# Split the documents into chunks
doc_splits = [Document(page_content=rule) for rule in rules_db]
print(f"Generated {len(doc_splits)} individual rule chunks")
