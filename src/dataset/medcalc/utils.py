from collections import Counter
from datasets import Dataset, concatenate_datasets
import numpy as np

def balance_dataset_by_category(
    dataset: Dataset,
    category_column: str = "category",
    target_count: int = None,
    seed: int = 42
) -> Dataset:
    """
    Balance a dataset by upsampling or downsampling each category to a target count.

    Args:
        dataset (Dataset): The input HuggingFace Dataset object.
        category_column (str): The name of the column containing category labels.
        target_count (int, optional): Target number of samples per category. If None,
                                      will use the median count as default.
        seed (int): Random seed for reproducibility.

    Returns:
        Dataset: A new dataset with balanced category distribution.
    """
    # Count samples per category
    counts = Counter(dataset[category_column])

    # Determine target count
    if target_count is None:
        target_count = int(np.median(list(counts.values())))

    # Collect balanced datasets
    category_datasets = {}
    for cat, count in counts.items():
        cat_dataset = dataset.filter(lambda x: x[category_column] == cat)

        if count < target_count:
            # Upsampling with replacement
            indices = np.random.choice(
                len(cat_dataset),
                size=target_count,
                replace=True
            ).tolist()
            balanced = cat_dataset.select(indices)
        # elif count > target_count:
        #     # Downsampling without replacement
        #     indices = np.random.choice(
        #         len(cat_dataset),
        #         size=target_count,
        #         replace=False
        #     ).tolist()
        #     balanced = cat_dataset.select(indices)
        else:
            # No resampling needed
            balanced = cat_dataset

        category_datasets[cat] = balanced

    # Merge and shuffle the final dataset
    balanced_dataset = concatenate_datasets(list(category_datasets.values()))
    balanced_dataset = balanced_dataset.shuffle(seed=seed)

    return balanced_dataset


def get_balanced_validation_set(dataset: Dataset, category_column: str, total_val_size: int) -> Dataset:
    """
    Extract a balanced validation set by taking min number of samples from each category.
    """
    # Count how many samples in each category
    category_counts = Counter(dataset[category_column])
    min_count = min(category_counts.values())
    
    # We will take at most min_count samples per class
    num_categories = len(category_counts)
    per_class = min(min_count, total_val_size // num_categories)

    val_indices = []
    used_indices = set()
    label_to_count = Counter()

    for idx, example in enumerate(dataset):
        label = example[category_column]
        if label_to_count[label] < per_class:
            val_indices.append(idx)
            label_to_count[label] += 1
            used_indices.add(idx)
        if sum(label_to_count.values()) >= per_class * num_categories:
            break

    val_data = dataset.select(val_indices)
    remaining_data = dataset.select([i for i in range(len(dataset)) if i not in used_indices])
    return val_data, remaining_data


def count_samples_per_category(dataset: Dataset, category_column: str = "category") -> dict:
    """
    Count the number of samples in each category.

    Args:
        dataset (Dataset): The input HuggingFace Dataset object.
        category_column (str): The name of the column containing category labels.

    Returns:
        dict: A dictionary mapping each category to its sample count.
    """
    return dict(Counter(dataset[category_column]))