import bisect
import hashlib
import logging
import random
from os.path import dirname, abspath, join

# Project root for this module (…/web_agent_site)
BASE_DIR = dirname(abspath(__file__))

# If you want to debug with a smaller number of products, set an int (e.g., 1000).
# Set to None to disable product-size limiting.
DEBUG_PROD_SIZE = None  # set to `None` to disable

# -----------------------------
# Data file paths (IMPORTANT)
# -----------------------------
# Switch from 1k subset -> full dataset:
#   items_ins_v2.json (attributes)
#   items_shuffle.json (product scraped info)
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')

# Reviews file (if present)
DEFAULT_REVIEW_PATH = join(BASE_DIR, '../data/reviews.json')

# Optional precomputed features (if your pipeline uses them)
FEAT_CONV = join(BASE_DIR, '../data/feat_conv.pt')
FEAT_IDS = join(BASE_DIR, '../data/feat_ids.pt')

# Human instruction/attribute data used by some components
HUMAN_ATTR_PATH = join(BASE_DIR, '../data/items_human_ins.json')


def random_idx(cum_weights):
    """
    Sample an index from a cumulative weight array.

    cum_weights:
        A list/array where cum_weights[i] is the cumulative sum up to i,
        and cum_weights is non-decreasing.

    Steps:
      1) Sample a uniform random number in [0, total_weight)
      2) Find insertion position with bisect (keeps sorted order)
      3) Clamp idx so we never return the last cumulative entry
         (common pattern when cum_weights includes an extra tail element)
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx


def setup_logger(session_id, user_log_dir):
    """
    Create a per-session logger that writes JSONL logs.

    session_id:
        Used as the logger name and filename.
    user_log_dir:
        A pathlib.Path directory where logs are stored.
    """
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def generate_mturk_code(session_id: str) -> str:
    """
    Generate a short redeem code for MTurk workers based on session_id.

    Uses SHA1(session_id) and returns the first 10 hex chars in uppercase.
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()