import ollama

db = []
try:
    with open('rules_rag/data/sample.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            clean_line = line.strip()
            if clean_line != '':
                db.append(line.strip())
        print(f'Loaded {len(db)} entries')
except FileNotFoundError:
    print("No such file.")
