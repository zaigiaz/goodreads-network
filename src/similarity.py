import numpy as np
import time


def build_genre_matrix(books):
    """One-hot encode all genres into a binary matrix."""
    print("  [1/3] Collecting all genres...")
    t0 = time.time()
    
    all_genres = set()
    for book in books.values():
        all_genres.update(book.get('genres', []))
    
    genre_list = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    
    print(f"  [2/3] Building matrix ({len(books)} books x {len(genre_list)} genres)...")
    n_books = len(books)
    n_genres = len(genre_list)
    
    genre_matrix = np.zeros((n_books, n_genres), dtype=np.float32)
    
    book_ids = list(books.keys())
    for i, book_id in enumerate(book_ids):
        for genre in books[book_id].get('genres', []):
            if genre in genre_to_idx:
                genre_matrix[i, genre_to_idx[genre]] = 1
        if i % 2000 == 0 and i > 0:
            print(f"    Progress: {i}/{n_books} books")
    
    print(f"  [3/3] Done. Time: {time.time() - t0:.2f}s")
    return genre_matrix, genre_list, book_ids


def compute_genre_similarity(genre_matrix):
    """Compute Jaccard similarity between all pairs of books."""
    print("  Computing genre similarity (matrix multiply)...")
    t0 = time.time()
    
    n = genre_matrix.shape[0]
    print(f"    Matrix size: {n}x{n}")
    
    intersection = genre_matrix @ genre_matrix.T
    print(f"    Intersection done")
    
    row_sums = genre_matrix.sum(axis=1)
    union = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - intersection
    
    similarity = np.zeros_like(intersection)
    nonzero = union > 0
    similarity[nonzero] = intersection[nonzero] / union[nonzero]
    
    np.fill_diagonal(similarity, 0)
    
    print(f"  Genre similarity done. Time: {time.time() - t0:.2f}s")
    return similarity


def compute_rating_similarity(ratings, max_diff=1.0):
    """Compute rating similarity for all pairs."""
    print("  Computing rating similarity...")
    t0 = time.time()
    
    diff = np.abs(ratings[:, np.newaxis] - ratings[np.newaxis, :])
    similarity = np.maximum(0, 1 - diff / max_diff)
    
    np.fill_diagonal(similarity, 0)
    
    print(f"  Rating similarity done. Time: {time.time() - t0:.2f}s")
    return similarity


def compute_combined_similarity(genre_sim, rating_sim, weights):
    """Combine genre and rating similarity with weights."""
    print("  Combining similarities...")
    t0 = time.time()
    
    genre_weight, rating_weight = weights
    total_weight = genre_weight + rating_weight
    
    combined = (genre_weight * genre_sim + rating_weight * rating_sim) / total_weight
    
    print(f"  Combined done. Time: {time.time() - t0:.2f}s")
    return combined


def build_adjacency_matrix(similarity_matrix, threshold):
    """Create adjacency (binary) matrix."""
    print(f"  Building adjacency (threshold={threshold})...")
    t0 = time.time()
    
    adj = (similarity_matrix >= threshold).astype(np.float32)
    edges = np.sum(np.triu(adj, k=1))
    
    print(f"  Adjacency done. {int(edges)} edges. Time: {time.time() - t0:.2f}s")
    return adj