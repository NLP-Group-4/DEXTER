#!/usr/bin/env python3
"""
Experiment 4: Hard Negative Injection (k = 1, 3, 5)
Retrieves semantically similar but incorrect documents (Hard Negatives) 
using Contriever and mixes them with Gold evidence.
"""

import os
import csv
import time
import random
import torch
from pathlib import Path

from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.utils.metrics.ExactMatch import ExactMatch

# --- CONFIGURATION ---
NOISE_LEVELS = [1, 3, 5]
SHUFFLE_CONTEXTS = True

# Initialize ExactMatch metric
em_metric = ExactMatch()
UNIQUE_QUESTIONS_LIMIT = 1200 
MAX_CORPUS_SIZE = 600000  
TOP_K_RETRIEVAL = 20     # We only need enough to fill k=5 noise slots

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
    output_dir = Path("results/Experiment_4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Gemma
    print("Loading Gemma 3 4B model via Ollama...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name="gemma3:4b", 
        temperature=0.3,
        max_new_tokens=256
    )

    # 2. Load Dataset
    print("Loading Dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(project_root, "evaluation", "config.ini")
    
    loader = RetrieverDataset(
        "wikimultihopqa", "wiki-musiqueqa-corpus",
        config_path, Split.DEV, tokenizer=None
    )
    
    # 3. Initialize Contriever
    print("Initializing Contriever...")
    retriever_config = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=32 
    )
    retriever = Contriever(retriever_config)
    similarity_measure = DotScore()

    # 4. Data Prep
    queries, qrels, corpus_list = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    # Efficient Question Filtering
    print("Filtering questions...")
    unique_ids = set()
    limited_raw_data = []
    
    for row in raw_data:
        qid = row.question.id()
        if qid not in unique_ids:
            if len(unique_ids) >= UNIQUE_QUESTIONS_LIMIT:
                pass 
            else:
                unique_ids.add(qid)
        if qid in unique_ids:
            limited_raw_data.append(row)
    
    raw_data = limited_raw_data
    print(f"Dataset prepared: {len(unique_ids)} unique questions.")

    # 5. Corpus Prep for Retrieval
    print(f"Preparing corpus (Sampling {MAX_CORPUS_SIZE} docs)...")
    if len(corpus_list) > MAX_CORPUS_SIZE:
        corpus_subset = random.sample(corpus_list, MAX_CORPUS_SIZE)
    else:
        corpus_subset = corpus_list
    
    print("Mining Hard Negatives via Contriever...")
    
    # Build list of unique Question objects for retrieval
    unique_questions = {}
    for row in raw_data:
        qid = row.question.id()
        if qid not in unique_questions:
            unique_questions[qid] = row.question
    
    target_queries = list(unique_questions.values())  # List of Question objects
    
    retrieval_results = retriever.retrieve(
        corpus_subset,
        target_queries,  # Now passing Question objects, not dict
        TOP_K_RETRIEVAL,
        similarity_measure
    )
    print("Hard Negative Mining Complete.")

    # 7. Main Experiment Loop
    for k_noise in NOISE_LEVELS:
        print(f"\n{'='*40}")
        print(f"STARTING RUN: {k_noise} HARD NEGATIVES")
        print(f"{'='*40}")
        
        output_file = output_dir / f"gemma3_hardneg_k{k_noise}.tsv"
        
        # Init CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["questions", "answers", "gold", "context_used"])
        
        matches = 0
        mismatches = 0
        processed_count = 0
        
        current_qid = None
        gold_evidences = []
        
        for index, row in enumerate(raw_data):
            qid = row.question.id()
            
            # --- Evidence Aggregation ---
            if current_qid is None: current_qid = qid
            
            if qid == current_qid:
                gold_evidences.append(row.evidences.text())
                is_last_item = (index + 1 >= len(raw_data))
                if not is_last_item and raw_data[index+1].question.id() == current_qid:
                    continue
            
            # --- HARD NEGATIVE SELECTION ---
            # Get docs retrieved by Contriever
            retrieved_map = retrieval_results.get(qid, {})
            
            # Create a set of gold texts for filtering
            gold_set = set(g.strip().lower() for g in gold_evidences)
            
            hard_negatives = []
            
            # Map retrieved IDs back to text
            # Note: retrieval_results keys are IDs, values are scores
            for doc_id in retrieved_map.keys():
                # Find doc in corpus
                doc_obj = next((d for d in corpus_subset if str(d.id()) == str(doc_id)), None)
                
                if doc_obj:
                    text = doc_obj.text()
                    # Ensure it's not the actual answer doc (Gold)
                    if text.strip().lower() not in gold_set:
                        hard_negatives.append(text)
                
                if len(hard_negatives) >= k_noise:
                    break
            
            # Fallback: If Contriever failed to find enough distinct docs, fill with randoms
            if len(hard_negatives) < k_noise:
                remaining_needed = k_noise - len(hard_negatives)
                random_fill = random.sample(corpus_subset, remaining_needed)
                for d in random_fill:
                    hard_negatives.append(d.text())

            # --- COMBINE & SHUFFLE ---
            combined_context = gold_evidences + hard_negatives
            if SHUFFLE_CONTEXTS:
                random.shuffle(combined_context)
            
            evidence_text = " ".join(combined_context)
            
            # --- INFERENCE ---
            user_prompt = PROMPT_TEMPLATE.format(
                evidence=evidence_text,
                question=row.question.text()
            )
            
            chain_answer = ""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chain_answer = llm_instance.get_gemma_completion("Follow instructions.", user_prompt)
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    time.sleep(2)
                    if attempt == max_retries - 1: chain_answer = "ERROR"
            
            # --- SCORING (Using ExactMatch from utils) ---
            extracted = chain_answer
            if "[Final Answer]:" in chain_answer:
                extracted = chain_answer.split("[Final Answer]:")[-1].strip()
            
            if em_metric.evaluate(extracted, row.answer.text()):
                matches += 1
            else:
                mismatches += 1
            
            # --- WRITE TO FILE ---
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                clean_q = row.question.text().replace('\t', ' ').replace('\n', ' ')
                clean_a = chain_answer.replace('\t', ' ').replace('\n', ' ')
                clean_g = row.answer.text().replace('\t', ' ').replace('\n', ' ')
                clean_c = evidence_text.replace('\t', ' ').replace('\n', ' ')[:5000]
                writer.writerow([clean_q, clean_a, clean_g, clean_c])
                
            processed_count += 1
            
            # Reset
            gold_evidences = []
            if not is_last_item:
                current_qid = raw_data[index+1].question.id()
            
            if processed_count % 50 == 0:
                print(f"k={k_noise} | EM: {matches/processed_count:.4f} | Processed: {processed_count}/1200")

        print(f"FINISHED k={k_noise} | Final EM: {matches/processed_count:.4f}")