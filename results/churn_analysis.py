import pandas as pd
import numpy as np
import os

ORACLE_PATH = 'results/Experiment_2/gemma3_oracle_retrieved_v2_ollama.tsv'
NOISE_FILES = {
    'k=1': 'results/Experiment_3/gemma3_noise_k1.tsv',
    'k=3': 'results/Experiment_3/gemma3_noise_k3.tsv',
    'k=5': 'results/Experiment_3/gemma3_noise_k5.tsv'
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

def is_correct(row, answer_col='answers', gold_col='gold'):
    """Checks if the gold answer is in the model's prediction."""
    try:
        ans = str(row[answer_col])
        gold = str(row[gold_col])
        
        # Check for refusal/uncertainty keywords if necessary (optional)
        if "not possible" in ans.lower() or "unknown" in ans.lower():
            return 0
            
        # Extract content after [Final Answer]:
        if "[Final Answer]:" in ans:
            pred = ans.split("[Final Answer]:")[-1].strip()
        else:
            pred = ans # Fallback if format is missing
            
        # Case-insensitive check
        return 1 if gold.lower() in pred.lower() else 0
    except:
        return 0

def run_comprehensive_analysis():
    print("Loading datasets...")
    
    # 1. Load Oracle (Baseline)
    if not os.path.exists(ORACLE_PATH):
        print(f"Error: Oracle file not found at {ORACLE_PATH}")
        return
        
    oracle_df = pd.read_csv(ORACLE_PATH, sep='\t')
    oracle_df['is_correct'] = oracle_df.apply(lambda r: is_correct(r, 'answers', 'gold'), axis=1)
    oracle_df['q_type'] = oracle_df['questions'].apply(categorize_question)
    
    # Store results for cross-level comparison
    # Structure: {'Oracle': [1, 0, 1...], 'k=1': [1, 1, 0...]}
    comparison_data = {
        'questions': oracle_df['questions'].values,
        'Oracle': oracle_df['is_correct'].values
    }
    
    # 2. Iterate through Noise Levels
    for label, file_path in NOISE_FILES.items():
        if not os.path.exists(file_path):
            print(f"Skipping {label}: File not found ({file_path})")
            continue
            
        print(f"\n{'='*40}")
        print(f"ANALYZING: {label}")
        print(f"{'='*40}")
        
        noise_df = pd.read_csv(file_path, sep='\t')
        
        # Merge to ensure alignment
        merged = pd.merge(oracle_df, noise_df, on='questions', suffixes=('_oracle', '_noise'))
        
        # Calculate Correctness for Noise
        merged['correct_noise'] = merged.apply(lambda r: is_correct(r, 'answers_noise', 'gold_noise'), axis=1)
        
        # Store for cross-comparison later
        comparison_data[label] = merged['correct_noise'].values
        
        # --- Part A: Churn Analysis (Table 2) ---
        # Define Status
        merged['status'] = 'Stable'
        merged.loc[(merged['is_correct']==1) & (merged['correct_noise']==0), 'status'] = 'Lost'
        merged.loc[(merged['is_correct']==0) & (merged['correct_noise']==1), 'status'] = 'Gained'
        
        # Counts
        lost_count = len(merged[merged['status'] == 'Lost'])
        gained_count = len(merged[merged['status'] == 'Gained'])
        stable_correct = len(merged[(merged['is_correct']==1) & (merged['correct_noise']==1)])
        stable_wrong = len(merged[(merged['is_correct']==0) & (merged['correct_noise']==0)])
        net_change = gained_count - lost_count
        
        print(f"\n[Overall Churn Stats for {label}]")
        print(f"Stable Correct: {stable_correct}")
        print(f"Stable Wrong:   {stable_wrong}")
        print(f"Lost (Destructive): {lost_count}")
        print(f"Gained (Beneficial): {gained_count}")
        print(f"Net Change: {net_change:+d}")
        
        # --- Part B: Question Type Breakdown (Table 3) ---
        breakdown = merged.groupby(['status', 'q_type']).size().unstack(fill_value=0)
        
        # Ensure Lost/Gained rows exist
        if 'Lost' not in breakdown.index: breakdown.loc['Lost'] = 0
        if 'Gained' not in breakdown.index: breakdown.loc['Gained'] = 0
        
        type_net = breakdown.loc['Gained'] - breakdown.loc['Lost']
        
        print(f"\n[Breakdown by Question Type for {label}]")
        print(breakdown)
        print("\n[Net Change per Type]")
        print(type_net)
        
        # --- Part C: Answer Length Analysis ---
        avg_len_oracle = merged['answers_oracle'].str.len().mean()
        avg_len_noise = merged['answers_noise'].str.len().mean()
        print(f"\n[Answer Length Analysis]")
        print(f"Oracle Avg Length: {avg_len_oracle:.2f} chars")
        print(f"Noise  Avg Length: {avg_len_noise:.2f} chars (Delta: {avg_len_noise - avg_len_oracle:+.2f})")

    # --- Part D: Robustness & Fragility (Cross-Level Analysis) ---
    print(f"\n{'='*40}")
    print("CROSS-LEVEL ROBUSTNESS ANALYSIS")
    print(f"{'='*40}")
    
    # Create DataFrame from stored results
    try:
        comp_df = pd.DataFrame(comparison_data)
        
        # Check alignment
        if len(comp_df) != len(oracle_df):
            print("Warning: Comparison DataFrame size mismatch. Some merges might have dropped rows.")
        
        # Define logic
        comp_df['All_Noise_Correct'] = (comp_df['k=1'] == 1) & (comp_df['k=3'] == 1) & (comp_df['k=5'] == 1)
        comp_df['All_Noise_Wrong']   = (comp_df['k=1'] == 0) & (comp_df['k=3'] == 0) & (comp_df['k=5'] == 0)
        
        # Metrics
        robust_correct = len(comp_df[(comp_df['Oracle'] == 1) & (comp_df['All_Noise_Correct'] == 1)])
        fragile = len(comp_df[(comp_df['Oracle'] == 1) & (comp_df['All_Noise_Wrong'] == 1)])
        miracle = len(comp_df[(comp_df['Oracle'] == 0) & (comp_df['All_Noise_Correct'] == 1)])
        
        # Any Correct (Ceiling)
        cols_to_check = ['Oracle'] + list(NOISE_FILES.keys())
        comp_df['Any_Correct'] = comp_df[cols_to_check].max(axis=1)
        total_solvable = comp_df['Any_Correct'].sum()
        
        print(f"Robust Correct (Always correct despite noise): {robust_correct} ({robust_correct/len(comp_df):.1%})")
        print(f"Fragile (Correct in Oracle, Wrong in ALL noise): {fragile}")
        print(f"Miracle (Wrong in Oracle, Correct in ALL noise): {miracle}")
        print(f"Total Unique Questions Solved at least once: {total_solvable}")
        
    except KeyError as e:
        print(f"\nCould not run cross-level analysis. Missing data for: {e}")

if __name__ == "__main__":
    run_comprehensive_analysis()