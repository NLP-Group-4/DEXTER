#!/usr/bin/env python3
"""
Experiment 3: Automated Noise Injection (k=1, 3, 5)
Mixes Oracle (Gold) contexts with Random Noise documents.
"""

import os
import csv
import time
import random
from pathlib import Path
from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.ExactMatch import ExactMatch

# --- CONFIGURATION ---
NOISE_LEVELS = [1, 3, 5]
SHUFFLE_CONTEXTS = True

# Initialize ExactMatch metric
em_metric = ExactMatch()

PROMPT_TEMPLATE = """[Question]: When does monsoon season end in the state the area code 575 is located?
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
[Final Answer]: February 4, 1948.

Follow the above example.
Given the evidence, Evidence: {evidence} 
use the information and answer the Question: {question}
Give answer strictly preceded by [Final Answer]:"""

if __name__ == "__main__":
    # Setup Output
    output_dir = Path("results/Experiment_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Model
    print("Loading Gemma 3 4B model via Ollama...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name="gemma3:4b", 
        temperature=0.3,
        max_new_tokens=256
    )
    
    # 2. Load Data
    print("Loading Dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(project_root, "evaluation", "config.ini")
    
    loader = RetrieverDataset(
        "wikimultihopqa", "wiki-musiqueqa-corpus",
        config_path, Split.DEV, tokenizer=None
    )
    queries, qrels, corpus_list = loader.qrels()
    raw_data = loader.base_dataset.raw_data

    # 3. Extract Corpus for Sampling
    print("Extracting corpus texts (this may take a moment)...")
    if hasattr(corpus_list[0], 'text'):
        all_corpus_texts = [doc.text() for doc in corpus_list]
    elif isinstance(corpus_list[0], dict) and 'text' in corpus_list[0]:
        all_corpus_texts = [doc['text'] for doc in corpus_list]
    else:
        all_corpus_texts = corpus_list
    print(f"Corpus loaded: {len(all_corpus_texts)} documents available for noise.")

    # 4. Filter to First 1200 Questions
    print("Filtering dataset...")
    unique_ids = set()
    limited_raw_data = []
    
    for row in raw_data:
        qid = row.question.id()
        if qid not in unique_ids:
            if len(unique_ids) >= 1200:
                pass # Continue to finish current ID evidences
            else:
                unique_ids.add(qid)
        if qid in unique_ids:
            limited_raw_data.append(row)
    
    raw_data = limited_raw_data
    print(f"Dataset prepared: {len(unique_ids)} unique questions.")

    # 5. Main Loop
    for k_noise in NOISE_LEVELS:
        print(f"\n{'='*40}")
        print(f"STARTING RUN: k={k_noise} NOISE DOCUMENTS")
        print(f"{'='*40}")
        
        output_file = output_dir / f"gemma3_noise_k{k_noise}.tsv"
        
        # Initialize File with Headers
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["questions", "answers", "gold", "context_used"])

        matches = 0
        mismatches = 0
        processed_count = 0
        
        current_qid = None
        gold_evidences = []

        for index, row in enumerate(raw_data):
            # --- Aggregate Evidence Logic ---
            if current_qid is None:
                current_qid = row.question.id()
            
            if row.question.id() == current_qid:
                gold_evidences.append(row.evidences.text())
                
                # Check if next row is same question. If so, continue accumulating.
                is_last_item = (index + 1 >= len(raw_data))
                if not is_last_item and raw_data[index+1].question.id() == current_qid:
                    continue
            
            # --- START PROCESSING QUESTION ---
            
            # 1. Noise Injection
            noise_docs = random.sample(all_corpus_texts, k_noise)
            combined_context = gold_evidences + noise_docs
            
            if SHUFFLE_CONTEXTS:
                random.shuffle(combined_context)
            
            evidence_text = " ".join(combined_context)
            
            # 2. Prepare Prompt
            user_prompt = PROMPT_TEMPLATE.format(
                evidence=evidence_text, 
                question=row.question.text()
            )
            
            # 3. Inference with Retry
            chain_answer = ""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chain_answer = llm_instance.get_gemma_completion(
                        "Follow instructions.", 
                        user_prompt
                    )
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    time.sleep(2)
                    if attempt == max_retries - 1:
                        chain_answer = "ERROR_FAILED"

            # 4. Robust Scoring (using ExactMatch from utils)
            extracted_answer = chain_answer
            if "[Final Answer]:" in chain_answer:
                extracted_answer = chain_answer.split("[Final Answer]:")[-1].strip()
            
            is_correct = em_metric.evaluate(extracted_answer, row.answer.text())
            if is_correct:
                matches += 1
            else:
                mismatches += 1

            # 5. Write to File
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                clean_q = row.question.text().replace('\t', ' ').replace('\n', ' ')
                clean_a = chain_answer.replace('\t', ' ').replace('\n', ' ')
                clean_g = row.answer.text().replace('\t', ' ').replace('\n', ' ')
                clean_c = evidence_text.replace('\t', ' ').replace('\n', ' ')[:5000] # Limit context log size
                writer.writerow([clean_q, clean_a, clean_g, clean_c])

            # 6. Reset for next
            gold_evidences = []
            if not is_last_item:
                current_qid = raw_data[index+1].question.id()
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"k={k_noise} | EM: {matches/processed_count:.4f} | Processed: {processed_count}/1200")

        print(f"FINISHED k={k_noise} | Final EM: {matches/processed_count:.4f}")

    print("\nALL EXPERIMENTS COMPLETE.")