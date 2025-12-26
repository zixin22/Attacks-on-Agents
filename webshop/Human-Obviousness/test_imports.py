"""
Quick test script to verify all imports work correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing imports...")
    
    # Test basic imports
    import json
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("✓ Basic imports successful")
    
    # Test attack module import
    from attack import FragmentAttackGenerator
    print("✓ Attack module import successful")
    
    # Test data file exists
    webshop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(webshop_dir, "data", "selected_reward_0.5", "extracted_goals.json")
    if os.path.exists(data_file):
        print(f"✓ Data file found: {data_file}")
    else:
        print(f"✗ Data file not found: {data_file}")
    
    print("\nAll imports successful! You can run analyze_human_obviousness.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install missing packages:")
    print("pip install matplotlib seaborn scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

