"""
Text Similarity Module
Various text similarity computation methods
"""

import os
# Set HuggingFace mirror for better connectivity
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from typing import List, Tuple, Union, Optional

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as st_cos_sim
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None
    st_cos_sim = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    TfidfVectorizer = None
    sklearn_cosine = None

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None


def levenshtein_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein edit distance (0-1, 1=identical)"""
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    max_len = max(m, n)
    return 1.0 - (dp[m][n] / max_len)


def levenshtein_distance_matrix(strings: List[str]) -> np.ndarray:
    """Compute Levenshtein distance matrix for string list"""
    n = len(strings)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance(strings[i], strings[j])
            matrix[i][j] = dist
            matrix[j][i] = dist
        matrix[i][i] = 1.0
    return matrix


def string_matching(s1: str, s2: str, method: str = 'jaccard') -> float:
    """Compute string matching similarity (jaccard/overlap/dice)"""
    if not s1 or not s2:
        return 0.0

    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if method == 'jaccard':
        return intersection / union if union > 0 else 0.0
    elif method == 'overlap':
        min_len = min(len(set1), len(set2))
        return intersection / min_len if min_len > 0 else 0.0
    elif method == 'dice':
        total = len(set1) + len(set2)
        return (2 * intersection) / total if total > 0 else 0.0
    else:
        return intersection / union if union > 0 else 0.0


def string_matching_matrix(strings: List[str], method: str = 'jaccard') -> np.ndarray:
    """Compute string matching similarity matrix for string list"""
    n = len(strings)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = string_matching(strings[i], strings[j], method)
            matrix[i][j] = sim
            matrix[j][i] = sim
        matrix[i][i] = 1.0
    return matrix


def cosine_similarity_e5(text_a: str, text_b: str,
                         embedding_model_name: str = 'intfloat/e5-base-v2') -> float:
    """Compute cosine similarity using e5-base-v2 embeddings"""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers not installed")

    if not text_a or not text_b:
        return 0.0

    query_a = f"query: {text_a}" if len(text_a) < 256 else text_a
    query_b = f"query: {text_b}" if len(text_b) < 256 else text_b

    model = SentenceTransformer(embedding_model_name)
    emb_a = model.encode(query_a, normalize_embeddings=True)
    emb_b = model.encode(query_b, normalize_embeddings=True)

    similarity = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, (similarity + 1) / 2))


def cosine_similarity_all_minilm(text_a: str, text_b: str,
                                 embedding_model_name: str = 'all-MiniLM-L6-v2') -> float:
    """Compute cosine similarity using all-MiniLM-L6-v2 embeddings"""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers not installed")

    if not text_a or not text_b:
        return 0.0

    model = SentenceTransformer(embedding_model_name)
    emb_a = model.encode(text_a, normalize_embeddings=True)
    emb_b = model.encode(text_b, normalize_embeddings=True)

    similarity = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, (similarity + 1) / 2))


def bm25_similarity(query: str, documents: List[str]) -> np.ndarray:
    """Compute BM25 similarity between query and document list"""
    if not HAS_BM25:
        raise ImportError("rank-bm25 not installed")

    if not query or not documents:
        return np.zeros(len(documents) if documents else 0)

    tokenized_query = query.lower().split()
    tokenized_docs = [doc.lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


if __name__ == "__main__":
    test_texts = [
        "i am looking for a wireless charging cradle",
        "i need a phone charger mount",
        "searching for mobile wireless charger",
        "find me a bicycle pump"
    ]

    print("=" * 60)
    print("Text Similarity Module Test")
    print("=" * 60)

    print("\n1. Levenshtein Distance:")
    dist = levenshtein_distance(test_texts[0], test_texts[1])
    print(f"   '{test_texts[0][:30]}...' vs '{test_texts[1][:30]}...': {dist:.4f}")

    print("\n2. String Matching (Jaccard):")
    sim = string_matching(test_texts[0], test_texts[1])
    print(f"   '{test_texts[0][:30]}...' vs '{test_texts[1][:30]}...': {sim:.4f}")

    print("\n3. String Matching Matrix:")
    matrix = string_matching_matrix(test_texts[:4])
    print(f"   Shape: {matrix.shape}")

    print("\n4. Cosine Similarity (all-MiniLM):")
    sim = cosine_similarity_all_minilm(test_texts[0], test_texts[1])
    print(f"   '{test_texts[0][:30]}...' vs '{test_texts[1][:30]}...': {sim:.4f}")

    print("\n5. BM25 Similarity:")
    scores = bm25_similarity("wireless charger", test_texts)
    print(f"   Query 'wireless charger': {scores}")

    print("\n" + "=" * 60)
