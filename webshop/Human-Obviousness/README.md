# Human-Obviousness Analysis

This folder contains code for analyzing the human-obviousness of fragment attack instructions and trigger instructions compared to original instructions.

## Overview

The analysis performs the following tasks:

1. **Load Instructions**: Reads the first 100 instructions from `extracted_goals.json`
2. **Generate Attack Instructions**: For each instruction, generates:
   - Fragment attack instruction (Group B)
   - Trigger attack instruction (Group C)
3. **Generate Embeddings**: Converts all instructions to embeddings using sentence transformers
4. **Calculate Perplexity**: Computes perplexity scores using GPT-2 model
5. **Visualize Distributions**: Creates visualizations comparing:
   - Embedding distributions (PCA and t-SNE)
   - Perplexity distributions
   - Length distributions (word count)
   - Box plots for comparison

## Files

- `analyze_human_obviousness.py`: Main analysis script
- `README.md`: This file

## Output Files

After running the script, the following files will be generated:

- `results.json`: Complete results with all instructions and their attack variants
- `statistics.json`: Summary statistics for each group
- `distribution_analysis.png`: Main visualization showing distributions
- `comparison_boxplots.png`: Box plots comparing groups

## Usage

```bash
cd webshop/Human-Obviousness
python analyze_human_obviousness.py
```

## Requirements

The script requires the following additional packages (beyond the base requirements):

```bash
pip install matplotlib seaborn scikit-learn
```

## Configuration

You can modify the following constants in the script:

- `NUM_INSTRUCTIONS`: Number of instructions to analyze (default: 100)
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
- `PPL_MODEL`: Language model for perplexity calculation (default: "gpt2")
- `DATA_FILE`: Path to extracted_goals.json

## Groups

- **Group A**: Original instructions
- **Group B**: Fragment attack instructions
- **Group C**: Trigger attack instructions

## Analysis Dimensions

1. **Embedding Space**: Uses PCA and t-SNE to visualize how attack instructions differ from originals in embedding space
2. **Perplexity**: Measures how "surprising" or unusual the text is to a language model
3. **Length**: Character and word count distributions

## Notes

- The script uses the same instruction as both host and target for fragment generation (for analysis purposes)
- Perplexity calculation may take some time for large datasets
- GPU is recommended for faster processing but not required

