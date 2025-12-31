import pandas as pd
import numpy as np

# Load the files
files = {
    "Oracle": "results/Experiment_2/gemma3_oracle_retrieved_v2_ollama.tsv",
    "Noise_k1": "results/Experiment_3/gemma3_noise_k1.tsv",
    "Noise_k3": "results/Experiment_3/gemma3_noise_k3.tsv",
    "Noise_k5": "results/Experiment_3/gemma3_noise_k5.tsv"
}

# Helper function to determine correctness per row
def is_correct(row):
    try:
        if pd.isna(row['answers']) or pd.isna(row['gold']):
            return 0
        
        chain_answer = str(row['answers'])
        gold_answer = str(row['gold'])
        
        # User script logic
        if "not possible" in chain_answer.lower():
            return 0
        elif "unknown" in chain_answer.lower():
            return 0
        elif "[Final Answer]:" in chain_answer:
            pred_answer = chain_answer.split("[Final Answer]:")[-1].strip()
            if gold_answer.lower() in pred_answer.lower():
                return 1
        return 0
    except:
        return 0

# Read and process
results = {}
for name, path in files.items():
    df = pd.read_csv(path, sep='\t')
    df['is_correct'] = df.apply(is_correct, axis=1)
    results[name] = df[['questions', 'is_correct', 'gold', 'answers']]

# Check alignment
ref_questions = results['Oracle']['questions'].values
for name in ['Noise_k1', 'Noise_k3', 'Noise_k5']:
    if not np.array_equal(results[name]['questions'].values, ref_questions):
        print(f"Warning: Questions in {name} do not strictly align with Oracle by index/text.")
    else:
        print(f"{name} aligns with Oracle.")

# Merge for comparison
# Base df
comparison = results['Oracle'][['questions', 'is_correct']].rename(columns={'is_correct': 'Oracle_Correct'})

for name in ['Noise_k1', 'Noise_k3', 'Noise_k5']:
    comparison[f'{name}_Correct'] = results[name]['is_correct']

# Analysis
analysis = {}

for k in ['Noise_k1', 'Noise_k3', 'Noise_k5']:
    # Lost: Oracle Correct (1) -> Noise Wrong (0)
    lost = len(comparison[(comparison['Oracle_Correct'] == 1) & (comparison[f'{k}_Correct'] == 0)])
    
    # Gained: Oracle Wrong (0) -> Noise Correct (1)
    gained = len(comparison[(comparison['Oracle_Correct'] == 0) & (comparison[f'{k}_Correct'] == 1)])
    
    # Stable Correct: 1 -> 1
    stable_correct = len(comparison[(comparison['Oracle_Correct'] == 1) & (comparison[f'{k}_Correct'] == 1)])
    
    # Stable Wrong: 0 -> 0
    stable_wrong = len(comparison[(comparison['Oracle_Correct'] == 0) & (comparison[f'{k}_Correct'] == 0)])
    
    analysis[k] = {
        "Lost (Destructive Noise)": lost,
        "Gained (Beneficial Noise)": gained,
        "Stable Correct": stable_correct,
        "Stable Wrong": stable_wrong,
        "Net Change": gained - lost
    }

print("\nDetailed Analysis:")
print(pd.DataFrame(analysis).T)

# Calculate average length of answer generation
for name, df in results.items():
    avg_len = df['answers'].str.len().mean()
    print(f"\n{name} Avg Answer Length: {avg_len:.2f} chars")

# Correctness Matrix
correctness_df = comparison.copy()
correctness_df['All_Noise_Correct'] = (correctness_df['Noise_k1_Correct'] == 1) & (correctness_df['Noise_k3_Correct'] == 1) & (correctness_df['Noise_k5_Correct'] == 1)
correctness_df['All_Noise_Wrong'] = (correctness_df['Noise_k1_Correct'] == 0) & (correctness_df['Noise_k3_Correct'] == 0) & (correctness_df['Noise_k5_Correct'] == 0)

# Robustness
robust_correct = len(correctness_df[(correctness_df['Oracle_Correct'] == 1) & (correctness_df['All_Noise_Correct'] == 1)])
fragile = len(correctness_df[(correctness_df['Oracle_Correct'] == 1) & (correctness_df['All_Noise_Wrong'] == 1)])

# Miracle cures (Wrong in Oracle, Correct in ALL noise settings - unlikely but interesting)
miracle = len(correctness_df[(correctness_df['Oracle_Correct'] == 0) & (correctness_df['All_Noise_Correct'] == 1)])

print(f"\nRobust Correct (Always correct despite noise): {robust_correct}")
print(f"\nFragile (Correct in Oracle, Wrong in ALL noise): {fragile}")
print(f"\nMiracle (Wrong in Oracle, Correct in ALL noise): {miracle}")

# How many questions are correct in at least one setting? (Potential Ceiling)
correctness_df['Any_Correct'] = correctness_df[['Oracle_Correct', 'Noise_k1_Correct', 'Noise_k3_Correct', 'Noise_k5_Correct']].max(axis=1)
print(f"\nTotal Unique Questions Solved at least once: {correctness_df['Any_Correct'].sum()}")