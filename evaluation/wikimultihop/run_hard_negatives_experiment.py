#!/usr/bin/env python3
"""
Experiment 4: Hard Negative Injection (k = 1, 3, 5)

This experiment evaluates RAG robustness when GOLD evidence is mixed with
HARD NEGATIVES retrieved via dense retrieval (Contriever).

Key Experimental Parameters:
--------------------------------------------------
TOP_K_RETRIEVAL        = 50     # Final number of documents retrieved per query
CANDIDATE_POOL_SIZE   = 100    # Number of candidates per query for hard-negative mining
MAX_QUESTIONS         = 1200   # Limit on unique questions evaluated
MAX_CORPUS_SIZE       = 10000   # Size of sampled corpus subset for efficiency

"""


import os
import random
import pandas as pd
from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.utils.metrics.SimilarityMatch import DotScore


# --- CONFIGURATION ---
NOISE_LEVELS = [1, 3, 5]  # The experiment will run 3 times, once for each level
SHUFFLE_CONTEXTS = True   # Essential to prevent positional bias
UNIQUE_QUESTIONS_LIMIT = 1200  # Limit to first 1200 unique questions
MAX_CORPUS_SIZE = 10000  # Max corpus size for retrieval
CANDIDATE_POOL_SIZE = 100  # Number of candidates to consider for hard negatives
TOP_K_RETRIEVAL = 50  # Number of documents to retrieve per query



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

    print("Initializing Contriever...")
    retriever_config = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=32
    )

    retriever = Contriever(retriever_config)
    similarity_measure = DotScore()

    
    # Extract corpus texts for fast sampling
    # We create a simple list of strings from the corpus to sample hard negatives from
    queries, qrels, corpus_list = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    print("Extracting corpus texts for sampling...")

    # Limit to first 1200 unique questions
    unique_question_ids = []
    limited_raw_data = []
    for row in raw_data:
        if row.question.id() not in unique_question_ids:
            if len(unique_question_ids) >= UNIQUE_QUESTIONS_LIMIT:
                break
            unique_question_ids.append(row.question.id())
        limited_raw_data.append(row)
    raw_data = limited_raw_data
    print(f"Dataset prepared: {len(unique_question_ids)} unique questions.")

    # Subsample corpus since retrieval over the full corpus is expensive
    corpus_subset = random.sample(corpus_list, MAX_CORPUS_SIZE)

    # Build a candidate pool for HARD NEGATIVES
    # For each query, retrieve top-100 documents
    print("Building candidate pool for hard negatives...")
    
    # Bootstrap retrieval over the sampled corpus
    # This step is ONLY for mining hard negatives (not final evaluation)
    bootstrap_results = retriever.retrieve(
        corpus_subset,
        queries,
        CANDIDATE_POOL_SIZE,
        similarity_measure
    )

    # Collect all document IDs retrieved for any query
    # These are likely hard negatives (high similarity)
    candidate_doc_ids = set()

    for qid, doc_scores in bootstrap_results.items():
        for doc_id in doc_scores.keys():
            candidate_doc_ids.add(str(doc_id))

    # Always include gold docs
    for row in raw_data:
        try:
            candidate_doc_ids.add(str(row.evidences.id()))
        except Exception:
            for ev in row.evidences:
                candidate_doc_ids.add(str(ev.id()))

    # Build the reduced corpus:
    # Only documents that were either:
    #   - retrieved during bootstrapping, or
    #   - explicitly gold evidence
    reduced_corpus = [d for d in corpus_subset if str(d.id()) in candidate_doc_ids]

    print(f"Reduced corpus size: {len(reduced_corpus)}")

    # Final retrieval results
    retrieval_results = retriever.retrieve(
        reduced_corpus,
        queries,
        TOP_K_RETRIEVAL,
        similarity_measure
    )


    # 3. Main Experiment Loop (Iterate through noise levels)
    for k_noise in NOISE_LEVELS:
        print(f"\n{'='*40}")
        print(f"STARTING RUN: {k_noise} HARD NEGATIVES")
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
            
            # Multi-hop logic: Since questions may span multiple rows (one for each gold 
            # evidence), we aggregate gold_evidences until we encounter a new question_id.
            if row.question.id() == current_question_id:
                gold_evidences.append(row.evidences.text())
                
                # If next row is same question, continue collecting (don't process yet)
                if index + 1 < len(raw_data) and raw_data[index+1].question.id() == current_question_id:
                    continue
            
            # --- HARD NEGATIVE INJECTION ---
            # 1. Extract documents retrieved by Contriever for this specific question
            retrieved_map = retrieval_results[row.question.id()]

            # 2. Filter: Keep retrieved docs ONLY if they aren't the ground truth (Gold)
            retrieved_docs = []
            for cid, score in retrieved_map.items():
                match = next((d for d in reduced_corpus if str(d.id()) == str(cid)), None)
                if match:
                    retrieved_docs.append(match.text())

            gold_set = set(g.strip().lower() for g in gold_evidences)

            hard_negatives = []
            for doc_text in retrieved_docs:
                if doc_text.strip().lower() not in gold_set:
                    hard_negatives.append(doc_text)
                if len(hard_negatives) == k_noise:
                    break

            # 3. Fallback: If Contriever didn't return enough non-gold docs, 
            # fill the remaining slots from the general retrieval pool
            if len(hard_negatives) < k_noise:
                fallback = [
                    t for t in retrieved_docs
                    if t.strip().lower() not in gold_set
                ]
                if fallback:
                    hard_negatives.extend(
                        random.sample(
                            fallback,
                            min(k_noise - len(hard_negatives), len(fallback))
                        )
                    )

            
            # Combine Gold + Hard Negatives
            combined_contexts = gold_evidences + hard_negatives
            
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
                print(f"HardNeg k={k_noise} | EM: {matches/total_processed:.4f} | Processed: {total_processed}/1200")
                # Save checkpoint specific to this noise level
                pd.DataFrame(question_df).to_csv(f"gemma3_noise_k{k_noise}.tsv", sep="\t", index=False)
            
            # Reset for next question
            gold_evidences = []
            if index + 1 < len(raw_data):
                current_question_id = raw_data[index+1].question.id()

        # Final Save for this level
        final_em = matches / len(question_df['questions'])
        print(f"FINISHED HardNeg k={k_noise} | Final EM: {final_em:.4f}")
        pd.DataFrame(question_df).to_csv(f"gemma3_hardneg_k{k_noise}.tsv", sep="\t", index=False)

    print("\nALL HARD NEGATIVE EXPERIMENTS COMPLETE.")

