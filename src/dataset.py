import os
from tqdm import tqdm
import random

def load_imdb_dataset(data_dir="data/aclImdb", split="train"):
    """
    Loads the IMDb dataset.

    Parameters
    ----------
    data_dir : str
        Root directory of the IMDb dataset
    split : str
        'train' or 'test'

    Returns
    -------
    texts : list
        List of review strings
    labels : list
        List of labels (1=positive, 0=negative)
    """

    texts = []
    labels = []

    pos_dir = os.path.join(data_dir, split, "pos")
    neg_dir = os.path.join(data_dir, split, "neg")

    # Positive reviews
    for filename in tqdm(os.listdir(pos_dir), desc="Loading positive reviews"):
        path = os.path.join(pos_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(1)

    # Negative reviews
    for filename in tqdm(os.listdir(neg_dir), desc="Loading negative reviews"):
        path = os.path.join(neg_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(0)

    combined = list(zip(texts, labels))
    random.shuffle(combined)

    texts, labels = zip(*combined)

    return list(texts), list(labels)
    