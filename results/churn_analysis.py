import pandas as pd
import os
from dexter.utils.metrics.ExactMatch import ExactMatch

# Initialize ExactMatch metric
em_metric = ExactMatch()

# --- CONFIGURATION ---
ORACLE_PATH = 'results/Experiment_2/gemma3_oracle_retrieved_v2_ollama.tsv'

# Define experiment groups
EXPERIMENTS = {
    "Experiment 3: Random Noise": {
        'k=1': 'results/Experiment_3/gemma3_noise_k1.tsv',
        'k=3': 'results/Experiment_3/gemma3_noise_k3.tsv',
        'k=5': 'results/Experiment_3/gemma3_noise_k5.tsv'
    },
    "Experiment 4: Hard Negatives": {
        'k=1': 'results/Experiment_4/gemma3_hardneg_k1.tsv',
        'k=3': 'results/Experiment_4/gemma3_hardneg_k3.tsv',
        'k=5': 'results/Experiment_4/gemma3_hardneg_k5.tsv'
    }
}

def categorize_question(question):
    """Categorizes a question based on its starting word."""
    q_lower = str(question).lower().strip()
    if q_lower.startswith("who"): return "Who"
    if q_lower.startswith("what"): return "What"
    if q_lower.startswith("where"): return "Where"
    if q_lower.startswith("when"): return "When"
    if q_lower.startswith("which"): return "Which"
    return "Other"

def calculate_exact_match(row, answer_col, gold_col):
    """Calculates Exact Match using the ExactMatch utility."""
    try:
        ans = str(row[answer_col])
        gold = str(row[gold_col])
        
        # Extract content after [Final Answer]:
        if "[Final Answer]:" in ans:
            pred = ans.split("[Final Answer]:")[-1].strip()
        else:
            pred = ans
            
        return 1 if em_metric.evaluate(pred, gold) else 0
    except:
        return 0

def analyze_experiment_group(group_name, file_map, oracle_df):
    """Runs the full analysis suite for a specific set of files (e.g., Random Noise or Hard Negs)."""
    
    print(f"\n{'#'*60}")
    print(f"### ANALYZING: {group_name} ###")
    print(f"{'#'*60}")

    # Store results for cross-level comparison within this experiment
    comparison_data = {
        'questions': oracle_df['questions'].values,
        'Oracle': oracle_df['is_correct'].values
    }

    # 1. Iterate through levels (k=1, k=3, k=5)
    for label, file_path in file_map.items():
        if not os.path.exists(file_path):
            print(f"Skipping {label}: File not found ({file_path})")
            continue
            
        print(f"\n--- Level: {label} ---")
        
        # Load and Merge
        exp_df = pd.read_csv(file_path, sep='\t')
        
        # Merge without renaming first - let suffixes handle duplicates
        merged = pd.merge(oracle_df, exp_df, on='questions', suffixes=('', '_exp'))
        
        # Determine the correct column names after merge
        # Oracle has 'answers' and 'gold', experiment may have same or different names
        oracle_answer_col = 'answers'
        exp_answer_col = 'answers_exp' if 'answers_exp' in merged.columns else 'answers'
        exp_gold_col = 'gold_exp' if 'gold_exp' in merged.columns else 'gold'
        
        # Calculate Exact Match for the experiment
        merged['em_exp'] = merged.apply(lambda r: calculate_exact_match(r, exp_answer_col, exp_gold_col), axis=1)
        
        # Store for robustness check (using EM now)
        comparison_data[label] = merged['em_exp'].values
        
        # --- EXACT MATCH SCORES ---
        em_score_exp = merged['em_exp'].mean()
        total_questions = len(merged)
        
        print(f"[Exact Match (EM) Score]")
        print(f"  {group_name} ({label}): {em_score_exp:.4f} ({int(em_score_exp * total_questions)}/{total_questions})")
        
        # --- A. Churn Analysis (based on EM) ---
        merged['status'] = 'Stable'
        merged.loc[(merged['is_correct']==1) & (merged['em_exp']==0), 'status'] = 'Lost'
        merged.loc[(merged['is_correct']==0) & (merged['em_exp']==1), 'status'] = 'Gained'
        
        stable_correct = len(merged[(merged['is_correct']==1) & (merged['em_exp']==1)])
        stable_wrong = len(merged[(merged['is_correct']==0) & (merged['em_exp']==0)])
        lost_count = len(merged[merged['status'] == 'Lost'])
        gained_count = len(merged[merged['status'] == 'Gained'])
        net = gained_count - lost_count
        
        print(f"[Churn Stats]")
        print(f"Stable Correct: {stable_correct} | Stable Wrong: {stable_wrong}")
        print(f"Lost (Destructive): {lost_count} | Gained (Beneficial): {gained_count}")
        print(f"Net Change: {net:+d}")
        
        # --- B. Question Type Breakdown ---
        breakdown = merged.groupby(['status', 'q_type']).size().unstack(fill_value=0)
        
        # Calculate Net Change per Type
        if 'Lost' not in breakdown.index: breakdown.loc['Lost'] = 0
        if 'Gained' not in breakdown.index: breakdown.loc['Gained'] = 0
        type_net = breakdown.loc['Gained'] - breakdown.loc['Lost']
        
        print(f"\n[Net Change by Question Type]")
        print(type_net)
        
        # --- C. Answer Length Analysis ---
        avg_len_oracle = merged[oracle_answer_col].astype(str).str.len().mean()
        avg_len_exp = merged[exp_answer_col].astype(str).str.len().mean()
        print(f"\n[Answer Length]")
        print(f"Oracle: {avg_len_oracle:.2f} | {group_name} ({label}): {avg_len_exp:.2f} (Delta: {avg_len_exp - avg_len_oracle:+.2f})")

    # 2. Cross-Level Robustness Analysis
    print(f"\n--- {group_name} ROBUSTNESS SUMMARY ---")
    try:
        comp_df = pd.DataFrame(comparison_data)
        
        # Define what constitutes "All Noise Correct" for this specific experiment group
        # Dynamically grab keys that start with 'k='
        k_cols = [k for k in file_map.keys() if k in comp_df.columns]
        
        if not k_cols:
            print("No data available for robustness analysis.")
            return

        comp_df['All_Levels_Correct'] = comp_df[k_cols].all(axis=1) # True if 1 in all k cols
        comp_df['All_Levels_Wrong']   = (comp_df[k_cols] == 0).all(axis=1) # True if 0 in all k cols
        
        # Robust Correct: Correct in Oracle AND Correct in ALL experiment levels
        robust_correct = len(comp_df[(comp_df['Oracle'] == 1) & (comp_df['All_Levels_Correct'] == True)])
        
        # Fragile: Correct in Oracle BUT Wrong in ALL experiment levels
        fragile = len(comp_df[(comp_df['Oracle'] == 1) & (comp_df['All_Levels_Wrong'] == True)])
        
        # Miracle: Wrong in Oracle BUT Correct in ALL experiment levels
        miracle = len(comp_df[(comp_df['Oracle'] == 0) & (comp_df['All_Levels_Correct'] == True)])
        
        print(f"Robust Correct (Always correct despite interference): {robust_correct} ({robust_correct/len(comp_df):.1%})")
        print(f"Fragile (Correct in Oracle, collapsed in ALL levels): {fragile}")
        print(f"Miracle (Wrong in Oracle, fixed in ALL levels): {miracle}")
        
    except Exception as e:
        print(f"Could not calculate robustness stats: {e}")

def run_comprehensive_analysis():
    global ORACLE_PATH  # Declare global at the start
    
    print("Loading Oracle Baseline...")
    
    if not os.path.exists(ORACLE_PATH):
        print(f"Error: Oracle file not found at {ORACLE_PATH}")
        # Try finding it in v2 folder just in case
        alt_path = 'results/Experiment_2/gemma3_oracle_retrieved_v2_ollama.tsv'
        if os.path.exists(alt_path):
            print(f"Found Oracle at alternative path: {alt_path}")
            ORACLE_PATH = alt_path
        else:
            return
        
    oracle_df = pd.read_csv(ORACLE_PATH, sep='\t')
    oracle_df['is_correct'] = oracle_df.apply(lambda r: calculate_exact_match(r, 'answers', 'gold'), axis=1)
    oracle_df['q_type'] = oracle_df['questions'].apply(categorize_question)
    
    # Display Oracle Baseline Scores
    print(f"\n{'='*60}")
    print(f"ORACLE BASELINE (Experiment 2)")
    print(f"{'='*60}")
    oracle_em = oracle_df['is_correct'].mean()
    total_oracle = len(oracle_df)
    print(f"Exact Match (EM): {oracle_em:.4f} ({int(oracle_em * total_oracle)}/{total_oracle})")
    
    # Run analysis for each defined experiment
    for group_name, file_map in EXPERIMENTS.items():
        analyze_experiment_group(group_name, file_map, oracle_df)

if __name__ == "__main__":
    run_comprehensive_analysis()