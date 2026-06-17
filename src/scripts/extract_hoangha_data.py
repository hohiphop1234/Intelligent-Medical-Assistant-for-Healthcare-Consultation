import pandas as pd
import json
import re
from pathlib import Path

def remove_think_tags(text):
    # Remove everything inside <think>...</think>
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

def main():
    risk_categories = [
        "safety",
        "interactions",
        "contraindications",
        "contraindication",
        "pregnancy",
        "overdose",
        "pediatric",
        "patient_query",
        "case_based",
        "edge_case",
    ]
    
    print("Loading parquet dataset...")
    df = pd.read_parquet('data/hoangha_medical_dataset/train.parquet')
    
    # Filter by category
    df_filtered = df[df['category'].isin(risk_categories)]
    print(f"Found {len(df_filtered)} records matching risk categories.")
    
    output_file = Path('data/rag_processed/rag_chunks_hoangha_risk.jsonl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df_filtered.iterrows():
            question = str(row.get('question', '')).strip()
            answer = str(row.get('answer', ''))
            answer_clean = remove_think_tags(answer)
            
            if not question or not answer_clean:
                continue
                
            content = f"Question: {question}\nAnswer: {answer_clean}"
            category = str(row.get('category', ''))
            
            chunk = {
                "source": "hoangha_medical_dataset",
                "title": f"Medical Case - {category}",
                "content": content,
                "topic_group": category,
                "language": "vi",
                "url": "local://hoangha_medical_dataset"
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"Successfully wrote {processed_count} chunks to {output_file}")

if __name__ == '__main__':
    main()
