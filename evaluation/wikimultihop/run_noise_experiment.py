#!/usr/bin/env python3
"""
Experiment 3: Automated Noise Injection (k=1, 3, 5)
Mixes Oracle (Gold) contexts with Random Noise documents.
Runs sequentially for different noise levels.
"""

import os
import random
import pandas as pd
from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

# --- CONFIGURATION ---
NOISE_LEVELS = [1, 3, 5]  # The experiment will run 3 times, once for each level
SHUFFLE_CONTEXTS = True   # Essential to prevent positional bias

if __name__ == "__main__":
    # 1. Initialize Gemma 3 4B model via Ollama
    print("Loading Gemma 3 4B model via Ollama...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name="gemma3:4b", 
        temperature=0.3,
        max_new_tokens=256
    )
    
    # 2. Load Dataset & Corpus
    print("Loading Dataset and Corpus...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(project_root, "evaluation", "config.ini")
    
    loader = RetrieverDataset(
        "wikimultihopqa", "wiki-musiqueqa-corpus",
        config_path, Split.DEV, tokenizer=None
    )
    
    # Extract corpus texts for fast sampling
    # We create a simple list of strings from the corpus to sample noise from
    queries, qrels, corpus_list = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    print("Extracting corpus texts for sampling...")
    # Adjust this based on your actual corpus object structure. 
    # Usually corpus_list items are dicts or objects with a 'text' or 'page_content' field.
    # If corpus_list is just strings, use: all_corpus_texts = corpus_list
    if hasattr(corpus_list[0], 'text'):
        all_corpus_texts = [doc.text() for doc in corpus_list]
    elif isinstance(corpus_list[0], dict) and 'text' in corpus_list[0]:
        all_corpus_texts = [doc['text'] for doc in corpus_list]
    else:
        # Fallback if corpus is already a list of strings
        all_corpus_texts = corpus_list

    # Limit to first 1200 unique questions
    unique_question_ids = []
    limited_raw_data = []
    for row in raw_data:
        if row.question.id() not in unique_question_ids:
            if len(unique_question_ids) >= 1200:
                break
            unique_question_ids.append(row.question.id())
        limited_raw_data.append(row)
    raw_data = limited_raw_data
    print(f"Dataset prepared: {len(unique_question_ids)} unique questions.")

    # 3. Main Experiment Loop (Iterate through noise levels)
    for k_noise in NOISE_LEVELS:
        print(f"\n{'='*40}")
        print(f"STARTING RUN: {k_noise} RANDOM NOISE DOCUMENTS")
        print(f"{'='*40}")
        
        matches = 0
        mismatches = 0
        question_df = {"questions": [], "answers": [], "gold": [], "context_used": [], "noise_level": []}
        
        current_question_id = None
        gold_evidences = []
        
        # Process questions
        for index, row in enumerate(raw_data):
            # --- Evidence Aggregation Logic ---
            if current_question_id is None:
                current_question_id = row.question.id()
            
            if row.question.id() == current_question_id:
                gold_evidences.append(row.evidences.text())
                
                # If next row is same question, continue collecting (don't process yet)
                if index + 1 < len(raw_data) and raw_data[index+1].question.id() == current_question_id:
                    continue
            
            # --- NOISE INJECTION ---
            # Sample k random documents from the corpus
            # We assume the corpus is large enough that collision with gold evidence is negligible
            noise_docs = random.sample(all_corpus_texts, k_noise)
            
            # Combine Gold + Noise
            combined_contexts = gold_evidences + noise_docs
            
            # Shuffle to ensure the model doesn't rely on gold being at the top
            if SHUFFLE_CONTEXTS:
                random.shuffle(combined_contexts)
                
            evidence_text = " ".join(combined_contexts)
            
            # --- INFERENCE ---
            user_prompt = """[Few-shot examples...]
[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
[Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay Packers.
[Final Answer]: Green Bay Packers.\n\n """ + \
            "Follow the above example and Given the evidence, Evidence: " + evidence_text + \
            " \n use the information and answer the Question: " + row.question.text() + \
            " Give answer strictly preceded by [Final Answer]:"
            
            chain_answer = llm_instance.get_gemma_completion(
                "Follow instructions and answer based on context.", 
                user_prompt
            )
            
            # --- SCORING ---
            gold_answer = row.answer.text()
            if "[Final Answer]:" in chain_answer:
                pred_answer = chain_answer.split("[Final Answer]:")[-1].strip()
                if gold_answer.lower() in pred_answer.lower():
                    matches += 1
                else:
                    mismatches += 1
            else:
                mismatches += 1
            
            # Log Data
            question_df["questions"].append(row.question.text())
            question_df["answers"].append(chain_answer)
            question_df["gold"].append(gold_answer)
            question_df["context_used"].append(evidence_text)
            question_df["noise_level"].append(k_noise)
            
            # Progress update
            total_processed = len(question_df['questions'])
            if total_processed % 50 == 0:
                print(f"Noise k={k_noise} | EM: {matches/total_processed:.4f} | Processed: {total_processed}/1200")
                # Save checkpoint specific to this noise level
                pd.DataFrame(question_df).to_csv(f"gemma3_noise_k{k_noise}.tsv", sep="\t", index=False)
            
            # Reset for next question
            gold_evidences = []
            if index + 1 < len(raw_data):
                current_question_id = raw_data[index+1].question.id()

        # Final Save for this level
        final_em = matches / len(question_df['questions'])
        print(f"FINISHED k={k_noise} | Final EM: {final_em:.4f}")
        pd.DataFrame(question_df).to_csv(f"gemma3_noise_k{k_noise}.tsv", sep="\t", index=False)

    print("\nALL EXPERIMENTS COMPLETE.")