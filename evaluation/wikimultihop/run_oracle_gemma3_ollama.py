#!/usr/bin/env python3
"""
Experiment 2: Oracle Contexts with Gemma 3 4B (Ollama Backend)
Process first 1200 questions from dev.json using oracle (ground truth) contexts
Uses Ollama backend for better Mac compatibility
"""

import os
import pandas as pd
from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from sklearn.metrics.pairwise import cosine_similarity
from dexter.config.constants import Split
from sentence_transformers import SentenceTransformer
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from torch import Tensor
from typing import List, Dict

"""
def get_top_k_similar_instances(
    sentence: str, data_emb: Tensor, data: List[Dict],
    k: int, threshold: float
) -> List[Dict]:
    sent_emb = model.encode(sentence)
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    results_sims = zip(range(len(text_sims)), text_sims)
    sorted_similarities = sorted(
        results_sims, key=lambda x: x[1], reverse=True)
    top_questions = []
    for idx, item in sorted_similarities[:k]:
        if item[0] > threshold:
            top_questions.append(list(data)[idx])
    return top_questions
"""

#model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")

if __name__ == "__main__":
    # Initialize Gemma 3 4B model via Ollama
    print("Loading Gemma 3 4B model via Ollama...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name="gemma3:4b",  # Ollama model name
        temperature=0.3,
        max_new_tokens=256
    )
    print("Model loaded")
    
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(project_root, "evaluation", "config.ini")
    
    loader = RetrieverDataset(
        "wikimultihopqa", "wiki-musiqueqa-corpus",
        config_path, Split.DEV, tokenizer=None
    )
    queries, qrels, corpus_list = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    # Limit to first 1200 questions
    print(f"Total questions in dataset: {len(set([row.question.id() for row in raw_data]))}")
    unique_question_ids = []
    limited_raw_data = []
    for row in raw_data:
        if row.question.id() not in unique_question_ids:
            if len(unique_question_ids) >= 1200:
                break
            unique_question_ids.append(row.question.id())
        if len(unique_question_ids) <= 1200:
            limited_raw_data.append(row)
    
    raw_data = limited_raw_data
    print(f"Processing first 1200 questions (total samples with evidences: {len(raw_data)})")
    
    system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"
    matches = 0
    mismatches = 0
    ids = []
    evidences = []
    question_df = {"questions": [], "answers": [], "gold": []}
    
    for index, row in enumerate(raw_data):
        # Check if this is the last item
        is_last_item = (index + 1 >= len(raw_data))
        # Check if next question is different (or if this is the last item)
        is_last_of_question = is_last_item or (row.question.id() != raw_data[index+1].question.id())

        if row.question.id() in ids and not is_last_of_question:
            evidences.append(row.evidences.text())
            continue
        elif row.question.id() in ids and is_last_of_question:
            evidences.append(row.evidences.text())
        elif row.question.id() not in ids and is_last_of_question:
            ids.append(row.question.id())
            evidences.append(row.evidences.text())
        elif row.question.id() not in ids and not is_last_of_question:
            ids.append(row.question.id())
            evidences = []
            evidences.append(row.evidences.text())
            continue
                
        # Process the question (this now runs for all questions including the last one)
        #evidence_emb = model.encode(evidences)
        #evidences_final = get_top_k_similar_instances(
        #    row.question.text(), evidence_emb, evidences, 3, 0.5
        #)
        evidence_text = " ".join(evidences)
        
        user_prompt = """[Question]: When does monsoon season end in the state the area code 575 is located?
[Answer]: The area code 575 is located in New Mexico. Monsoon season in New Mexico typically ends in mid-September. So the
[Final Answer]: mid-September.
[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
[Answer]: Ineabelle Diaz is from Peurto Rico, which is in the United States of America. The current official currency in the United
States is the United States dollar. 
[Final Answer]: United States dollar.
[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
[Answer]: The person who founded the American Institute of Public Opinion in 1935 is George Gallup. George Gallup was born
in Jefferson, Iowa. 
[Final Answer]: Jefferson.
[Question]: What language is used by the director of Tiffany Memorandum?
[Answer]: The director of Tiffany Memorandum is Sergio Grieco. Sergio Grieco speaks Italian.
[Final Answer]: Italian.
[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
[Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay
Packers.
[Final Answer]: Green Bay Packers.
[Question]: The birth country of Jayantha Ketagoda left the British Empire when?
[Answer]: The birth country of Jayantha Ketagoda is Sri Lanka. Sri Lanka left the British Empire on February 4, 1948. So the
[Final Answer]: February 4, 1948.\n\n """ + "Follow the above example and Given the evidence, Evidence: " + evidence_text + " \n use the information and answer the Question:" + row.question.text() + "Give answer strictly preceded by [Final Answer]:"
        
        chain_answer = llm_instance.get_gemma_completion(system_prompt, user_prompt)
        
        if "not possible" in chain_answer.lower():
            mismatches += 1
            evidences = []
            continue
        elif "unknown" in chain_answer.lower():
            mismatches += 1
            evidences = []
            continue
        elif len(chain_answer.split("[Final Answer]:")) > 1:
            answer = chain_answer.split("[Final Answer]:")[-1]
            print("************", answer, row.answer.text())
            if row.answer.text().lower() in answer.lower():
                matches += 1
            else:
                mismatches += 1
        else:
            mismatches += 1
        
        question_df["answers"].append(chain_answer)
        question_df["questions"].append(row.question.text())
        question_df["gold"].append(row.answer.text())
        
        final_questions = pd.DataFrame(question_df)
        print(f"EM {matches/(matches+mismatches):.4f} | Processed: {len(question_df['questions'])}/1200")
        final_questions.to_csv("gemma3_oracle_retrieved_v2_ollama.tsv", sep="\t", index=False)
        
        # Reset evidences for next question
        evidences = []
        
        # Checkpoint every 50 questions
        if len(question_df['questions']) % 50 == 0:
            print(f"Checkpoint: {len(question_df['questions'])} questions processed")

print(f"\nCOMPLETE! Processed {len(question_df['questions'])} questions")
print(f"Final EM: {matches/(matches+mismatches):.4f}")
