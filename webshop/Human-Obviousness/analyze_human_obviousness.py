"""
Human-Obviousness Analysis Script

This script:
1. Reads first 100 instructions from extracted_goals.json
2. Generates fragment attack instructions and trigger instructions for each
3. Converts to embeddings
4. Calculates perplexity
5. Visualizes distributions (embedding/PPL/length dimensions)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import attack module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack import FragmentAttackGenerator

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PPL_MODEL = "gpt2"  # Use GPT-2 for perplexity calculation
NUM_INSTRUCTIONS = 100
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the webshop directory (parent of Human-Obviousness)
WEBSHOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(WEBSHOP_DIR, "data", "selected_reward_0.5", "extracted_goals.json")


class PerplexityCalculator:
    """Calculate perplexity using a language model"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        print(f"Loading perplexity model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a text string.
        
        Args:
            text: Input text string
            
        Returns:
            Perplexity score (lower is better/more predictable)
        """
        if not text or not text.strip():
            return float('inf')
        
        try:
            # Tokenize input
            encodings = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = encodings.input_ids.to(self.device)
            
            # Calculate negative log likelihood
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss.item()
            
            # Perplexity is exp(negative log likelihood)
            perplexity = np.exp(neg_log_likelihood)
            
            return perplexity
        except Exception as e:
            print(f"Error calculating perplexity for text '{text[:50]}...': {e}")
            return float('inf')


def load_instructions(data_file: str, num_instructions: int = 100) -> List[Tuple[str, str]]:
    """
    Load first N instructions from extracted_goals.json
    
    Args:
        data_file: Path to extracted_goals.json
        num_instructions: Number of instructions to load
        
    Returns:
        List of (index, instruction) tuples
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading instructions from {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get first N instructions (sorted by key as integer)
    items = sorted([(int(k), v) for k, v in data.items()], key=lambda x: x[0])
    instructions = items[:num_instructions]
    
    print(f"Loaded {len(instructions)} instructions")
    return instructions


def generate_attack_instructions(instructions: List[Tuple[str, str]]) -> List[Dict]:
    """
    Generate fragment attack and trigger instructions for each instruction.
    
    Args:
        instructions: List of (index, instruction) tuples
        
    Returns:
        List of dictionaries with:
        - 'index': Original index
        - 'original': Original instruction (Group A)
        - 'fragment_attack': Fragment attack instruction (Group B)
        - 'trigger_attack': Trigger attack instruction (Group C)
        - 'fragments': List of fragments
    """
    print("\nGenerating attack instructions...")
    attack_generator = FragmentAttackGenerator(verbose=False)
    
    results = []
    
    for idx, instruction in instructions:
        print(f"Processing instruction {idx}: {instruction[:60]}...")
        
        # Use the instruction itself as target (for fragment attack generation)
        # In real attack, target would be different, but for analysis we use same
        target_instruction = instruction
        
        # Generate attack plan
        attack_plan = attack_generator.generate_attack_plan(
            host_instruction=instruction,
            target_instruction=target_instruction
        )
        
        # Get fragment attack instruction (use first one if multiple exist)
        # Note: Normally each instruction should have only one fragment attack instruction
        fragment_attacks = attack_plan.get('fragment_attacks', [])
        fragment_attack_instruction = None
        if fragment_attacks:
            # Use the first fragment attack instruction (normally there should be only one)
            fragment_attack_instruction = fragment_attacks[0]['instruction']
        else:
            # Fallback: create a simple fragment attack using the first fragment
            fragments = attack_plan.get('fragments', [])
            if fragments:
                # Find first non-empty fragment
                for fragment in fragments:
                    if fragment and fragment.strip():
                        fragment_attack_instruction = f"{instruction}. Fragment relates to {fragment}."
                        break
            if not fragment_attack_instruction:
                # Final fallback: use original instruction
                fragment_attack_instruction = instruction
        
        # Get trigger attack instruction
        trigger_attack_instruction = attack_plan.get('trigger_instruction', instruction)
        
        results.append({
            'index': str(idx),
            'original': instruction,
            'fragment_attack': fragment_attack_instruction,
            'trigger_attack': trigger_attack_instruction,
            'fragments': attack_plan.get('fragments', [])
        })
    
    print(f"Generated attack instructions for {len(results)} instructions")
    return results


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: SentenceTransformer model
        
    Returns:
        numpy array of embeddings (n_texts, embedding_dim)
    """
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def calculate_perplexities(texts: List[str], ppl_calculator: PerplexityCalculator) -> List[float]:
    """
    Calculate perplexities for a list of texts.
    
    Args:
        texts: List of text strings
        ppl_calculator: PerplexityCalculator instance
        
    Returns:
        List of perplexity scores
    """
    print(f"Calculating perplexities for {len(texts)} texts...")
    perplexities = []
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(texts)}")
        ppl = ppl_calculator.calculate_perplexity(text)
        perplexities.append(ppl)
    return perplexities


def calculate_lengths(texts: List[str]) -> List[int]:
    """Calculate character and word lengths for texts"""
    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]
    return char_lengths, word_lengths


def visualize_distributions(results: List[Dict], embeddings: Dict[str, np.ndarray], 
                            perplexities: Dict[str, List[float]], 
                            lengths: Dict[str, Tuple[List[int], List[int]]],
                            output_dir: str):
    """
    Visualize distributions for embedding, perplexity, and length dimensions.
    
    Args:
        results: List of result dictionaries
        embeddings: Dictionary mapping group names to embedding arrays
        perplexities: Dictionary mapping group names to perplexity lists
        lengths: Dictionary mapping group names to (char_lengths, word_lengths) tuples
        output_dir: Output directory for plots
    """
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Embedding visualization (using PCA or t-SNE for dimensionality reduction)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for embedding visualization
    all_embeddings = []
    all_labels = []
    for group_name in ['A', 'B', 'C']:
        if group_name in embeddings:
            all_embeddings.append(embeddings[group_name])
            all_labels.extend([group_name] * len(embeddings[group_name]))
    
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
        
        # PCA visualization
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(combined_embeddings)
        
        ax = axes[0, 0]
        for group_name in ['A', 'B', 'C']:
            if group_name in embeddings:
                mask = np.array(all_labels) == group_name
                ax.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          label=f'Group {group_name}', alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Embedding Distribution (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # t-SNE visualization (if dataset is not too large)
        if len(combined_embeddings) <= 300:
            ax = axes[0, 1]
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings) - 1))
            tsne_embeddings = tsne.fit_transform(combined_embeddings)
            
            for group_name in ['A', 'B', 'C']:
                if group_name in embeddings:
                    mask = np.array(all_labels) == group_name
                    ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                              label=f'Group {group_name}', alpha=0.6, s=50)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('Embedding Distribution (t-SNE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax = axes[0, 1]
            ax.text(0.5, 0.5, 't-SNE skipped\n(too many samples)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Embedding Distribution (t-SNE)')
    
    # 2. Perplexity distribution
    ax = axes[1, 0]
    for group_name in ['A', 'B', 'C']:
        if group_name in perplexities:
            ppl_values = perplexities[group_name]
            # Filter out infinite values
            ppl_values = [p for p in ppl_values if p != float('inf')]
            if ppl_values:
                ax.hist(ppl_values, bins=30, alpha=0.6, label=f'Group {group_name}', density=True)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title('Perplexity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Length distribution (word count)
    ax = axes[1, 1]
    for group_name in ['A', 'B', 'C']:
        if group_name in lengths:
            word_lengths = lengths[group_name][1]
            ax.hist(word_lengths, bins=20, alpha=0.6, label=f'Group {group_name}', density=True)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Density')
    ax.set_title('Length Distribution (Word Count)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # 4. Box plots for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Perplexity box plot
    ax = axes[0]
    box_data = []
    box_labels = []
    for group_name in ['A', 'B', 'C']:
        if group_name in perplexities:
            ppl_values = [p for p in perplexities[group_name] if p != float('inf')]
            if ppl_values:
                box_data.append(ppl_values)
                box_labels.append(f'Group {group_name}')
    if box_data:
        ax.boxplot(box_data, labels=box_labels)
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity Comparison')
        ax.grid(True, alpha=0.3)
    
    # Word length box plot
    ax = axes[1]
    box_data = []
    box_labels = []
    for group_name in ['A', 'B', 'C']:
        if group_name in lengths:
            word_lengths = lengths[group_name][1]
            box_data.append(word_lengths)
            box_labels.append(f'Group {group_name}')
    if box_data:
        ax.boxplot(box_data, labels=box_labels)
        ax.set_ylabel('Word Count')
        ax.set_title('Length Comparison')
        ax.grid(True, alpha=0.3)
    
    # Embedding distance box plot (distance from Group A centroid)
    ax = axes[2]
    if 'A' in embeddings and len(embeddings['A']) > 0:
        # Calculate centroid of Group A
        centroid_A = np.mean(embeddings['A'], axis=0)
        
        box_data = []
        box_labels = []
        for group_name in ['A', 'B', 'C']:
            if group_name in embeddings:
                distances = []
                for emb in embeddings[group_name]:
                    dist = np.linalg.norm(emb - centroid_A)
                    distances.append(dist)
                box_data.append(distances)
                box_labels.append(f'Group {group_name}')
        
        if box_data:
            ax.boxplot(box_data, labels=box_labels)
            ax.set_ylabel('Distance from Group A Centroid')
            ax.set_title('Embedding Distance Comparison')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_boxplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to {output_path}")
    plt.close()


def save_results(results: List[Dict], embeddings: Dict[str, np.ndarray],
                perplexities: Dict[str, List[float]], 
                lengths: Dict[str, Tuple[List[int], List[int]]],
                output_dir: str):
    """Save all results to JSON files"""
    print("\nSaving results...")
    
    # Save main results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_file}")
    
    # Save statistics
    stats = {
        'num_instructions': len(results),
        'perplexity_stats': {},
        'length_stats': {},
        'embedding_stats': {}
    }
    
    for group_name in ['A', 'B', 'C']:
        if group_name in perplexities:
            ppl_values = [p for p in perplexities[group_name] if p != float('inf')]
            if ppl_values:
                stats['perplexity_stats'][group_name] = {
                    'mean': float(np.mean(ppl_values)),
                    'std': float(np.std(ppl_values)),
                    'min': float(np.min(ppl_values)),
                    'max': float(np.max(ppl_values)),
                    'median': float(np.median(ppl_values))
                }
        
        if group_name in lengths:
            word_lengths = lengths[group_name][1]
            stats['length_stats'][group_name] = {
                'mean_words': float(np.mean(word_lengths)),
                'std_words': float(np.std(word_lengths)),
                'min_words': int(np.min(word_lengths)),
                'max_words': int(np.max(word_lengths))
            }
        
        if group_name in embeddings:
            emb_array = embeddings[group_name]
            stats['embedding_stats'][group_name] = {
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in emb_array])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in emb_array]))
            }
    
    stats_file = os.path.join(output_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to {stats_file}")


def main():
    """Main execution function"""
    print("="*80)
    print("Human-Obviousness Analysis")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load instructions
    instructions = load_instructions(DATA_FILE, NUM_INSTRUCTIONS)
    
    # Step 2: Generate attack instructions
    results = generate_attack_instructions(instructions)
    
    # Step 3: Prepare texts for each group
    group_A_texts = [r['original'] for r in results]
    group_B_texts = [r['fragment_attack'] for r in results]
    group_C_texts = [r['trigger_attack'] for r in results]
    
    # Step 4: Generate embeddings
    print("\n" + "="*80)
    print("Generating Embeddings")
    print("="*80)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = {
        'A': generate_embeddings(group_A_texts, embedding_model),
        'B': generate_embeddings(group_B_texts, embedding_model),
        'C': generate_embeddings(group_C_texts, embedding_model)
    }
    
    # Step 5: Calculate perplexities
    print("\n" + "="*80)
    print("Calculating Perplexities")
    print("="*80)
    ppl_calculator = PerplexityCalculator(PPL_MODEL)
    perplexities = {
        'A': calculate_perplexities(group_A_texts, ppl_calculator),
        'B': calculate_perplexities(group_B_texts, ppl_calculator),
        'C': calculate_perplexities(group_C_texts, ppl_calculator)
    }
    
    # Step 6: Calculate lengths
    print("\n" + "="*80)
    print("Calculating Lengths")
    print("="*80)
    lengths = {
        'A': calculate_lengths(group_A_texts),
        'B': calculate_lengths(group_B_texts),
        'C': calculate_lengths(group_C_texts)
    }
    
    # Step 7: Visualize
    print("\n" + "="*80)
    print("Visualizing Results")
    print("="*80)
    visualize_distributions(results, embeddings, perplexities, lengths, OUTPUT_DIR)
    
    # Step 8: Save results
    save_results(results, embeddings, perplexities, lengths, OUTPUT_DIR)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    for group_name in ['A', 'B', 'C']:
        print(f"\nGroup {group_name}:")
        if group_name in perplexities:
            ppl_values = [p for p in perplexities[group_name] if p != float('inf')]
            if ppl_values:
                print(f"  Perplexity - Mean: {np.mean(ppl_values):.2f}, Std: {np.std(ppl_values):.2f}")
        if group_name in lengths:
            word_lengths = lengths[group_name][1]
            print(f"  Word Count - Mean: {np.mean(word_lengths):.2f}, Std: {np.std(word_lengths):.2f}")
        if group_name in embeddings:
            emb_norms = [np.linalg.norm(emb) for emb in embeddings[group_name]]
            print(f"  Embedding Norm - Mean: {np.mean(emb_norms):.2f}, Std: {np.std(emb_norms):.2f}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

