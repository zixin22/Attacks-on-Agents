"""
Radar Chart for Retrieval Match Rate (Memory Size = 32)
Each agent as a separate chart with 4 methods on the axes
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data for memory size = 32
agents = {
    'RAP': {
        'BM25': 100.0,
        'String matching': 100.0,
        'Cosine similarity (all-MiniLM)': 100.0,
        'Cosine similarity (e5-base-v2)': 100.0
    },
    'SeeAct': {
        'BM25': 100.0,
        'String matching': 100.0,
        'Cosine similarity (all-MiniLM)': 100.0,
        'Cosine similarity (e5-base-v2)': 100.0
    },
    'OSAgent': {
        'BM25': 94.0,
        'String matching': 100.0,
        'Cosine similarity (all-MiniLM)': 100.0,
        'Cosine similarity (e5-base-v2)': 92.0
    },
    'InspAgent': {
        'BM25': 100.0,
        'String matching': 100.0,
        'Cosine similarity (all-MiniLM)': 100.0,
        'Cosine similarity (e5-base-v2)': 100.0
    }
}

# Define colors for each agent
colors = {
    'RAP': '#FF6B6B',        # Coral red
    'SeeAct': '#4ECDC4',     # Teal
    'OSAgent': '#45B7D1',    # Sky blue
    'InspAgent': '#96CEB4'   # Sage green
}

# Metrics labels (these will be the 4 corners of the radar)
categories = [
    'BM25',
    'String matching',
    'Cosine similarity (all-MiniLM)',
    'Cosine similarity (e5-base-v2)'
]

N = len(categories)

# Calculate angles for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Generate one radar chart per agent
for agent_name in agents.keys():
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Get values for this agent
    values = [agents[agent_name][cat] for cat in categories]
    values_plot = values + values[:1]

    # Plot the agent's radar chart
    ax.plot(angles, values_plot, 'o-', linewidth=5, label=agent_name,
            color=colors[agent_name], markersize=20)
    ax.fill(angles, values_plot, alpha=0.2, color=colors[agent_name])

    # Set the labels at the 4 corners (shortened names)
    ax.set_xticks(angles[:-1])
    display_labels = [
        'BM25',
        'String matching',
        'MiniLM-L6-v2',
        'e5-base-v2'
    ]
    ax.set_xticklabels(display_labels, fontsize=36, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=32, fontweight='bold')

    # Add gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Adjust label position for better visibility
    ax.tick_params(axis='y', pad=15)
    ax.tick_params(axis='x', pad=25)

    plt.tight_layout()

    # Save figure
    agent_clean = agent_name.replace(' ', '_')
    output_path = f'retrieval_{agent_clean}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = f'retrieval_{agent_clean}.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    plt.close()

print("\nAll 4 radar charts generated successfully!")
