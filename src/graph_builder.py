import networkx as nx
import numpy as np
import time
from data_loader import load_goodreads_data

from similarity import (
    build_genre_matrix,
    compute_genre_similarity,
    compute_rating_similarity,
    compute_num_ratings_similarity,
    compute_combined_similarity,
    build_adjacency_matrix,
)

from data_analysis import get_graph_stats

DATA_DIR = '../data'


def numpy_to_networkx(adj_matrix, book_ids, books, similarity_matrix=None):
    """Convert adjacency matrix to NetworkX graph."""
    print("  Converting to NetworkX...")
    t0 = time.time()
    n = len(book_ids)
    
    print("    Creating graph from numpy array...")
    t1 = time.time()
    G = nx.from_numpy_array(adj_matrix)
    print(f"    Graph created. Time: {time.time() - t1:.2f}s")
    
    if similarity_matrix is not None:
        print("    Adding edge weights...")
        t1 = time.time()
        total = n * (n - 1) // 2
        edges_added = 0
        for i, j in zip(*np.triu_indices(n, k=1)):
            if adj_matrix[i, j] > 0:
                G[i][j]['weight'] = float(similarity_matrix[i, j])
                edges_added += 1
        print(f"    Added {edges_added} weights. Time: {time.time() - t1:.2f}s")
    
    print("    Relabeling nodes...")
    t1 = time.time()
    mapping = {i: book_ids[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    print(f"    Relabeled. Time: {time.time() - t1:.2f}s")
    
    print("    Setting node attributes...")
    t1 = time.time()
    node_attrs = {
        book_id: {
            'title': str(book.get('title', '')),
            'author': str(book.get('author', '')),
            'avg_rating': float(book.get('avg_rating', 0)),
            'genres': str(','.join(book.get('genres', []))),
        }
        for book_id, book in books.items()
    }
    nx.set_node_attributes(G, node_attrs)
    print(f"    Attributes set. Time: {time.time() - t1:.2f}s")
    
    print(f"  NetworkX conversion done. Total: {time.time() - t0:.2f}s")
    return G


def compare_erdos_renyi(G, n_samples=5, compute_clustering=False):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n < 2:
        return {'nodes': n, 'edges': 0, 'avg_clustering': 0}
    
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        m = max_edges
    
    print("  Generating Erdős–Rényi graphs...")
    t0 = time.time()
    
    if not compute_clustering:
        print("    (skipped - use --clustering to enable)")
        return {
            'nodes': n,
            'edges': m,
            'avg_clustering': 0,
        }
    
    clustering_vals = []
    for i in range(n_samples):
        ER = nx.gnm_random_graph(n, m)
        try:
            clustering_vals.append(nx.average_clustering(ER))
            print(f"    Sample {i+1}/{n_samples} done")
        except:
            pass
    
    avg_clustering = sum(clustering_vals) / len(clustering_vals) if clustering_vals else 0
    print(f"  ER done. Time: {time.time() - t0:.2f}s")
    
    return {
        'nodes': n,
        'edges': m,
        'avg_clustering': avg_clustering,
    }


def build_graph_fast(books, similarity_threshold=0.7, weights=(0.6, 0.2, 0.2)):
    """Build graph using numpy for speed."""
    print("\n=== Building Graph ===")
    
    t_total = time.time()
    
    print("\n[1/8] Building genre matrix...")
    genre_matrix, genre_list, book_ids = build_genre_matrix(books)
    print(f"  Matrix shape: {genre_matrix.shape}, {len(genre_list)} unique genres")
    
    print("\n[2/8] Computing genre similarity...")
    genre_sim = compute_genre_similarity(genre_matrix)
    
    print("\n[3/8] Computing num_ratings similarity...")
    num_ratings = np.array([books[bid].get('num_ratings', 0) for bid in book_ids], dtype=np.float32)
    num_ratings_sim = compute_num_ratings_similarity(num_ratings)
    
    print("\n[4/8] Computing rating similarity...")
    ratings = np.array([books[bid].get('avg_rating', 0) for bid in book_ids], dtype=np.float32)
    rating_sim = compute_rating_similarity(ratings)
    
    print("\n[5/8] Combining similarities...")
    combined_sim = compute_combined_similarity(genre_sim, num_ratings_sim, rating_sim, weights)
    
    print("\n[6/8] Building adjacency matrix...")
    adj_matrix = build_adjacency_matrix(combined_sim, similarity_threshold)
    
    print("\n[7/8] Converting to NetworkX...")
    G = numpy_to_networkx(adj_matrix, book_ids, books, combined_sim)
    
    print("\n[8/8] Saving GML...")
    nx.write_gml(G, f'{DATA_DIR}/goodreads_network.gml')
    print(f"  Saved to {DATA_DIR}/goodreads_network.gml")
    
    print(f"\nGraph complete. Total build time: {time.time() - t_total:.2f}s")
    
    return G


def generate_er_network(n, m, save_path=None):
    print("\n=== Generating ER Network ===")
    t0 = time.time()
    
    ER = nx.gnm_random_graph(n, m)
    print(f"  Generated ER graph (n={n}, m={m})")
    
    if save_path:
        nx.write_gml(ER, save_path)
        print(f"  Saved to {save_path}")
    
    print(f"  Done. Time: {time.time() - t0:.2f}s")
    return ER


def export_gml(G, filepath):
    print(f"Exporting to GML ({G.number_of_edges()} edges)...")
    t0 = time.time()
    nx.write_gml(G, filepath)
    print(f"Exported. Time: {time.time() - t0:.2f}s")


def test_thresholds(books, weights, thresholds):
    results = []
    for thresh in thresholds:
        print(f"\n=== Threshold: {thresh} ===")
        G = build_graph_fast(books, similarity_threshold=thresh, weights=weights)
        stats = get_graph_stats(G)
        print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}, Density: {stats['density']:.4f}")
        print(f"Avg Degree: {stats['avg_degree']:.2f}, Max Degree: {stats['max_degree']}")
        print(f"Avg Clustering: {stats['avg_clustering']:.4f}")
        
        erdos_stats = compare_erdos_renyi(G)
        print(f"[ER] Clustering: {erdos_stats['avg_clustering']:.4f}")
        
        results.append((thresh, stats, erdos_stats))
    return results


if __name__ == '__main__':
    import sys
    
    weights = (0.6, 0.2, 0.2)
    compute_clustering = '--clustering' in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-er':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        m = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        if n is None or m is None:
            G = nx.read_gml(f'{DATA_DIR}/goodreads_network.gml')
            n = G.number_of_nodes()
            m = G.number_of_edges()
            print(f"Using n={n}, m={m} from existing graph")
        
        save_path = f'{DATA_DIR}/er_network.gml'
        ER = generate_er_network(n, m, save_path=save_path)
        print(f"ER network saved to {save_path}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--from-gml':
        filepath = sys.argv[2] if len(sys.argv) > 2 else f'{DATA_DIR}/goodreads_network.gml'
        print(f"Loading from {filepath}...")
        G = nx.read_gml(filepath)
        
        stats = get_graph_stats(G, compute_clustering=True)
        
        print(f"\n=== Graph Stats ===")
        print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
        print(f"Density: {stats['density']:.4f}, Avg Degree: {stats['avg_degree']:.2f}")
        if stats['avg_clustering']:
            print(f"Avg Clustering: {stats['avg_clustering']:.4f}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--test-thresholds':
        print("Loading data...")
        t0 = time.time()
        books = load_goodreads_data(f'{DATA_DIR}/goodreads_data.csv')
        print(f"Loaded {len(books)} books. Time: {time.time() - t0:.2f}s")
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        results = test_thresholds(books, weights, thresholds)
        print("\n=== Summary ===")
        print("Threshold | Our Edges | Our Clustering | ER Clustering")
        for thresh, stats, erdos in results:
            c = f"{stats['avg_clustering']:.4f}" if stats['avg_clustering'] else "skipped"
            print(f"  {thresh}     |  {stats['edges']:5d}   |   {c}    |   {erdos['avg_clustering']:.4f}")
    
    else:
        print("Loading data...")
        t0 = time.time()
        books = load_goodreads_data(f'{DATA_DIR}/goodreads_data.csv')
        print(f"Loaded {len(books)} books. Time: {time.time() - t0:.2f}s")
        
        G = build_graph_fast(books, similarity_threshold=0.7, weights=weights)
        
        stats = get_graph_stats(G, compute_clustering=compute_clustering)
        erdos_stats = compare_erdos_renyi(G, compute_clustering=compute_clustering)
        
        print(f"\n=== Our Graph (threshold=0.7) ===")
        print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
        print(f"Density: {stats['density']:.4f}, Avg Degree: {stats['avg_degree']:.2f}")
        if stats['avg_clustering'] is not None:
            print(f"Avg Clustering: {stats['avg_clustering']:.4f}")
        
        print(f"\n=== Erdős–Rényi (same n,m) ===")
        print(f"Avg Clustering: {erdos_stats['avg_clustering']:.4f}")
