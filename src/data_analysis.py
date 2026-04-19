import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.optimize import curve_fit
from networkx.algorithms import community
import time
import argparse

GML_FILES = {
    1: '../data/goodreads_network.gml',
    2: '../data/er_network.gml',
}

GML_TITLES = {
    1: 'Goodreads Network',
    2: 'ER Network',
}

IMG_DIR = '../img'


def get_graph_stats(G, compute_clustering=False):
    print("  Computing graph stats...")
    t0 = time.time()
    
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    print(f"    Nodes: {stats['nodes']}, Edges: {stats['edges']}, Density: {stats['density']:.4f}")
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = sum(degrees) / len(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        print(f"    Avg degree: {stats['avg_degree']:.2f}")
    
    if compute_clustering and G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        print("    Computing clustering coefficient (slow)...")
        try:
            stats['avg_clustering'] = nx.average_clustering(G)
            print(f"    Clustering: {stats['avg_clustering']:.4f}")
        except Exception as e:
            stats['avg_clustering'] = 0
            print(f"    Clustering failed: {e}")
    else:
        stats['avg_clustering'] = None
        print("    Skipping clustering (use --clustering to enable)")
    
    print(f"  Stats done. Time: {time.time() - t0:.2f}s")
    return stats


def compute_modularity(G, method='greedy_modularity'):
    print(f"  Computing modularity ({method})...")
    t0 = time.time()
    
    if method == 'greedy_modularity':
        communities = community.greedy_modularity_communities(G)
    elif method == 'louvain':
        communities = community.louvain_communities(G)
    elif method == 'label_propagation':
        communities = community.label_propagation_communities(G)
    else:
        communities = community.greedy_modularity_communities(G)
    
    num_communities = len(communities)
    print(f"    Found {num_communities} communities")
    
    try:
        mod = community.modularity(G, communities)
        print(f"    Modularity: {mod:.4f}")
    except Exception as e:
        mod = 0
        print(f"    Modularity failed: {e}")
    
    print(f"  Modularity done. Time: {time.time() - t0:.2f}s")
    return mod, communities


def detect_communities(G, methods=None):
    if methods is None:
        methods = ['greedy_modularity', 'louvain', 'label_propagation']
    
    results = {}
    
    for method in methods:
        print(f"\n=== Community Detection: {method} ===")
        try:
            mod, communities = compute_modularity(G, method=method)
            results[method] = {
                'modularity': mod,
                'num_communities': len(communities),
                'communities': communities,
            }
        except Exception as e:
            print(f"    Failed: {e}")
            results[method] = {'modularity': 0, 'num_communities': 0, 'communities': []}
    
    return results


def compute_assortativity(G):
    print("  Computing assortativity...")
    t0 = time.time()
    
    try:
        ass = nx.degree_assortativity_coefficient(G)
        print(f"    Assortativity: {ass:.4f}")
    except Exception as e:
        ass = 0
        print(f"    Assortativity failed: {e}")
    
    print(f"  Assortativity done. Time: {time.time() - t0:.2f}s")
    return ass


def get_degree_sequence(G):
    return np.array([d for _, d in G.degree()])


def fit_gamma(x, a, b):
    return b * (x ** (a - 1)) * np.exp(-x / b)


def plot_degree_distribution(G, er_G=None, title="Degree Distribution", save_path=None):
    degrees = get_degree_sequence(G)
    min_deg = max(1, int(degrees.min()))
    max_deg = int(degrees.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(degrees, bins=range(min_deg, max_deg + 2), 
                                   edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Degree (k)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{title}\nDegree Histogram', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    ax2.scatter(unique_degrees, counts, s=30, alpha=0.7, color='steelblue', label='Data')
    
    if len(unique_degrees) > 3:
        try:
            popt, _ = curve_fit(fit_gamma, unique_degrees.astype(float), 
                               counts.astype(float), p0=[2.0, degrees.mean()],
                               maxfev=5000)
            x_fit = np.linspace(min_deg, max_deg, 200)
            y_fit = fit_gamma(x_fit, *popt)
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Gamma fit: α={popt[0]:.2f}, θ={popt[1]:.2f}')
            ax2.legend(fontsize=10)
        except:
            ax2.legend(fontsize=10)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (k)', fontsize=12)
    ax2.set_ylabel('Frequency P(k)', fontsize=12)
    ax2.set_title(f'{title}\nLog-Log Degree Distribution', fontsize=14)
    ax2.grid(which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path.replace('../', '')}")
    
    plt.show()
    
    return fig


def compare_with_er_network(G, n_samples=5, save_path=None):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n < 2:
        print("Graph too small for comparison")
        return
    
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        m = max_edges
    
    print(f"Generating {n_samples} Erdős–Rényi networks (n={n}, m={m})...")
    
    er_sequences = []
    for i in range(n_samples):
        ER = nx.gnm_random_graph(n, m)
        er_sequences.append(get_degree_sequence(ER))
    
    er_degrees_mean = np.mean(er_sequences, axis=0)
    
    our_degrees = get_degree_sequence(G)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    min_deg = max(1, int(min(our_degrees.min(), er_degrees_mean.min())))
    max_deg = int(max(our_degrees.max(), er_degrees_mean.max()))
    
    ax1.hist(our_degrees, bins=range(min_deg, max_deg + 2), alpha=0.7, 
            edgecolor='black', color='steelblue', label='Our Network')
    ax1.hist(er_degrees_mean, bins=range(min_deg, max_deg + 2), alpha=0.5, 
            edgecolor='black', color='orange', label='ER Network (mean)')
    ax1.set_xlabel('Degree (k)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Degree Histogram Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    
    our_unique, our_counts = np.unique(our_degrees, return_counts=True)
    ax2.scatter(our_unique, our_counts, s=30, alpha=0.7, color='steelblue', label='Our Network')
    
    er_unique, er_counts = np.unique(er_degrees_mean.astype(int), return_counts=True)
    ax2.scatter(er_unique, er_counts, s=30, alpha=0.7, color='orange', marker='s', label='ER Network')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (k)', fontsize=12)
    ax2.set_ylabel('Frequency P(k)', fontsize=12)
    ax2.set_title('Log-Log Degree Distribution Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path.replace('../', '')}")
    
    plt.show()
    
    return fig


def analyze_network(filepath=None, title="Network", compute_clustering=False, compare_er=False, 
                     compute_communities=False, compute_assort=False):
    print("Loading graph...")
    t0 = open(filepath, 'r').close() or 0
    
    if filepath:
        G = nx.read_gml(filepath)
    else:
        G = nx.read_gml('../data/goodreads_network.gml')
    
    stats = get_graph_stats(G, compute_clustering=compute_clustering)
    
    print(f"\n=== {title} Statistics ===")
    print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    print(f"Density: {stats['density']:.6f}")
    print(f"Avg Degree: {stats['avg_degree']:.2f}")
    print(f"Max Degree: {stats['max_degree']}")
    print(f"Min Degree: {stats['min_degree']}")
    
    degrees = get_degree_sequence(G)
    mean_deg = degrees.mean()
    var_deg = degrees.var()
    print(f"Degree variance: {var_deg:.2f}")
    print(f"Degree std dev: {np.sqrt(var_deg):.2f}")
    
    if compute_assort:
        ass = compute_assortativity(G)
        print(f"Assortativity: {ass:.4f}")
    
    if compute_communities:
        print("\n=== Community Detection ===")
        results = detect_communities(G)
        for method, data in results.items():
            print(f"  {method}: {data['num_communities']} communities, Q={data['modularity']:.4f}")
    
    if compare_er:
        compare_with_er_network(G, n_samples=5, save_path=f'{IMG_DIR}/degree_comparison.png')
    else:
        plot_degree_distribution(G, title=title, save_path=f'{IMG_DIR}/degree_distribution.png')
    
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze degree distribution')
    parser.add_argument('--file', type=int, default=1,
                       help='Which GML file to load: 1=Goodreads, 2=ER (default: 1)')
    parser.add_argument('--compare-er', action='store_true',
                       help='Compare with ER network')
    parser.add_argument('--clustering', action='store_true',
                       help='Compute clustering coefficient')
    parser.add_argument('--communities', action='store_true',
                       help='Detect communities and compute modularity')
    parser.add_argument('--assortativity', action='store_true',
                       help='Compute assortativity')
    args = parser.parse_args()
    
    filepath = GML_FILES.get(args.file, GML_FILES[1])
    title = GML_TITLES.get(args.file, GML_TITLES[1])
    
    analyze_network(filepath, title=title, compute_clustering=args.clustering, compare_er=args.compare_er,
                    compute_communities=args.communities, compute_assort=args.assortativity)
