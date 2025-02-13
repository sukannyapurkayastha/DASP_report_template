
# Function to compute cosine similarity between embeddings
def compute_embedding_similarity(prediction, label):
    """
    Computes the cosine similarity between the embeddings of a predicted text and a reference label.

    Args:
        prediction (str): The predicted text.
        label (str): The reference label for comparison.

    Returns:
        float: Cosine similarity score between the two embeddings (range: -1 to 1).
    """
    pred_embedding = model.encode([prediction])
    label_embedding = model.encode([label])
    return cosine_similarity(pred_embedding, label_embedding)[0][0]

# Function to compute Levenshtein (edit) distance normalized by length
def compute_edit_distance(prediction, label):
    """
    Computes a normalized similarity score based on Levenshtein (edit) distance.

    The score is computed as:
        similarity = 1 - (edit_distance / max_length)
    where `max_length` is the length of the longer string to normalize the distance.

    Args:
        prediction (str): The predicted text.
        label (str): The reference label for comparison.

    Returns:
        float: Normalized similarity score (range: 0 to 1, where 1 means identical strings).
    """
    edit_dist = Levenshtein.distance(prediction, label)
    max_len = max(len(prediction), len(label))
    return 1 - (edit_dist / max_len)  # Normalize and return similarity

# Apply embedding similarity and edit distance to the DataFrame
df['embedding_similarity'] = df.apply(
    lambda row: compute_embedding_similarity(row['pred'], row['descs']),
    axis=1
)

df['edit_similarity'] = df.apply(
    lambda row: compute_edit_distance(row['pred'], row['descs']),
    axis=1
)

# Display results
print(df.head())