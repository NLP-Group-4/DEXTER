#!/usr/bin/env python3
"""
Experiment 2: Oracle Contexts with Gemma 3 4B (Ollama Backend)
Process first 1200 questions from dev.json using oracle (ground truth) contexts
"""

import os
import csv
import time
from pathlib import Path
from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.ExactMatch import ExactMatch

# Initialize ExactMatch metric
em_metric = ExactMatch()

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("results/Experiment_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gemma3_oracle_retrieved_v2_ollama.tsv"
    
    # Initialize Gemma 3 4B model via Ollama
    print("Loading Gemma 3 4B model via Ollama...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name="gemma3:4b", 
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
    
    print(f"Total questions in dataset: {len(set([row.question.id() for row in raw_data]))}")
    unique_question_ids = set()
    limited_raw_data = []
    
    for row in raw_data:
        qid = row.question.id()
        # Add ID if we haven't reached the limit
        if qid not in unique_question_ids:
            if len(unique_question_ids) >= 1200:
                pass # Don't break yet, we need to finish the current ID's evidences if any
            else:
                unique_question_ids.add(qid)
        
        # If this row belongs to one of our selected 1200 questions, keep it
        if qid in unique_question_ids:
            limited_raw_data.append(row)
            
    raw_data = limited_raw_data
    print(f"Processing first 1200 questions (total samples with evidences: {len(raw_data)})")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["questions", "answers", "gold"])

    system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"
    
    matches = 0
    mismatches = 0
    ids = []
    evidences = []
    processed_count = 0

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
                
        # Process the question
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
        
        chain_answer = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chain_answer = llm_instance.get_gemma_completion(system_prompt, user_prompt)
                break # Success
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(2) # Cooldown
                if attempt == max_retries - 1:
                    chain_answer = "ERROR_FAILED_INFERENCE"

        # 1. Validation checks
        if "not possible" in chain_answer.lower() or "unknown" in chain_answer.lower():
            mismatches += 1
            extracted_answer = chain_answer # Save full string for debugging
        elif "[Final Answer]:" in chain_answer:
            extracted_answer = chain_answer.split("[Final Answer]:")[-1].strip()
            
            # Use ExactMatch metric from utils
            if em_metric.evaluate(extracted_answer, row.answer.text()):
                matches += 1
            else:
                mismatches += 1
        else:
            mismatches += 1
            extracted_answer = chain_answer

        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            # Clean strings to prevent CSV breakage
            clean_q = row.question.text().replace('\t', ' ').replace('\n', ' ')
            clean_a = chain_answer.replace('\t', ' ').replace('\n', ' ')
            clean_g = row.answer.text().replace('\t', ' ').replace('\n', ' ')
            writer.writerow([clean_q, clean_a, clean_g])
        
        processed_count += 1
        total_so_far = matches + mismatches
        if total_so_far > 0:
            print(f"EM {matches/total_so_far:.4f} | Processed: {processed_count}/1200")

        # Reset evidences
        evidences = []
        
        if processed_count % 50 == 0:
            print(f"Checkpoint: {processed_count} questions processed")

print(f"\nCOMPLETE! Processed {processed_count} questions")
if (matches + mismatches) > 0:
    print(f"Final EM: {matches/(matches+mismatches):.4f}")