from langchain_community.document_loaders.csv_loader import CSVLoader

csv_file_path = "rules_rag/data/cleaned_cards.csv"
loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')

documents = loader.load()

print(type(documents[0]))
