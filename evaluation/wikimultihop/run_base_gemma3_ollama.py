import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import pandas as pd
import time
from pathlib import Path

from dexter.llms.gemma_ollama_engine import GemmaOllamaEngine
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.ExactMatch import ExactMatch

# ================= CONFIGURATION =================
# Change these values to run different experiments for the assignment
TOP_K = 1  # Set to 1, 3, or 5
USE_ORACLE = False  # Set False for RQ1 (Retrieved), True for RQ3 (Gold/Oracle)
MODEL_NAME = "gemma3:4b"
# =================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    context_path = os.path.join(project_root, "experiment_contexts.json")

    print(f"Initializing {MODEL_NAME} via GemmaOllamaEngine...")
    llm_instance = GemmaOllamaEngine(
        data="",
        model_name=MODEL_NAME,
        temperature=0.3,
        max_new_tokens=256
    )
    print("Model loaded")

    em_metric = ExactMatch()

    # 3. Load Contexts
    print(f"Loading contexts from {context_path}...")
    try:
        with open(context_path, "r") as f:
            raw_context_data = json.load(f)
        context_map = {str(item['id']): item for item in raw_context_data}
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {context_path}")
        exit(1)

    # 4. Load Dataset Questions
    config_path = os.path.join(project_root, "evaluation", "config.ini")
    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", config_path, Split.DEV)
    raw_data = loader.base_dataset.raw_data

    # 5. Define Prompts
    system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"

    few_shot_examples = """[Question]: When does monsoon season end in the state the area code 575 is located?
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
[Final Answer]: February 4, 1948.\n\n"""

    # 6. Evaluation Loop
    question_df = {"questions": [], "answers": [], "gold": []}
    matches = 0
    mismatches = 0
    processed_ids = set()

    print(f"Starting Generation. Mode: {'Oracle' if USE_ORACLE else 'Retrieved'} | K: {TOP_K}")

    output_dir = Path("results/My_Experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_str = "oracle" if USE_ORACLE else "retrieved"
    output_filename = f"results/My_Experiment/llama3_rag_k{TOP_K}_{mode_str}.tsv"

    for i, row in enumerate(raw_data):
        if len(processed_ids) >= 1200:
            break

        qid = str(row.question.id())
        if qid in processed_ids:
            continue

        if qid not in context_map:
            continue

        processed_ids.add(qid)

        # Get relevant docs based on config
        entry = context_map[qid]
        if USE_ORACLE:
            docs = entry['oracle_contexts']
        else:
            docs = entry['retrieved_contexts']

        current_docs = docs[:TOP_K]
        evidence_text = " ".join(current_docs)

        user_prompt = f"{few_shot_examples}Follow the above example and Given the evidence, Evidence: {evidence_text} \n use the information and answer the Question:{row.question.text()}" + "Give answer strictly preceded by [Final Answer]:"

        # Inference Call
        chain_answer = ""
        try:
            chain_answer = llm_instance.get_gemma_completion(system_prompt, user_prompt)
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            chain_answer = "ERROR"

        chain_answer_lower = chain_answer.lower()

        # Check for failure keywords
        if "not possible" in chain_answer_lower or "unknown" in chain_answer_lower:
            mismatches += 1
        elif "[final answer]:" in chain_answer_lower:
            try:
                extracted_answer = chain_answer.split("[Final Answer]:")[-1].strip()

                # Use the imported ExactMatch metric
                if em_metric.evaluate(extracted_answer, row.answer.text()):
                    matches += 1
                else:
                    mismatches += 1
            except Exception:
                mismatches += 1
        else:
            mismatches += 1

        # Logging
        clean_q = row.question.text().replace('\t', ' ').replace('\n', ' ')
        clean_a = chain_answer.replace('\t', ' ').replace('\n', ' ')
        clean_g = row.answer.text().replace('\t', ' ').replace('\n', ' ')

        question_df["questions"].append(clean_q)
        question_df["answers"].append(clean_a)
        question_df["gold"].append(clean_g)

        if len(processed_ids) % 10 == 0:
            acc = matches / (matches + mismatches)
            print(f"Processed: {len(processed_ids)} | Strict Accuracy: {acc:.4f}")

    # 7. Final Save
    final_questions = pd.DataFrame(question_df)
    acc = matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    print(f"FINAL STRICT EM: {acc:.4f}")

    final_questions.to_csv(output_filename, sep="\t", index=False)
    print(f"Saved results to {output_filename}")