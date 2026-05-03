"""
Improved Drug Interaction Network Visualization
- Clean cluster-separated interactive network
- Community detection
- PageRank sizing
- Clearer visualization
"""

import torch
import networkx as nx
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import plotly.graph_objects as go
    import community as community_louvain
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Install required packages:")
    print("pip install plotly python-louvain")


class DrugNetworkVisualizer:

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.graph = None
        self.drug_names = {}

    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------
    def load_data_from_pytorch(self):
        print("Loading graph from PyTorch file...")

        graph_path = self.data_dir / "drug_graph.pt"
        saved_data = torch.load(graph_path, weights_only=False)

        graph_data = saved_data["graph_data"]
        idx_to_drug = saved_data["idx_to_drug"]

        edge_index = graph_data.edge_index
        num_nodes = graph_data.x.shape[0]

        G = nx.Graph()

        for i in range(num_nodes):
            name = idx_to_drug.get(i, f"Drug_{i}")
            self.drug_names[i] = str(name)
            G.add_node(i)

        edges = edge_index.cpu().numpy()
        for i in range(edges.shape[1]):
            src, dst = int(edges[0, i]), int(edges[1, i])
            if src < dst:
                G.add_edge(src, dst)

        self.graph = G

        print(f"Graph loaded: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
        return G

    # ------------------------------------------------------------------
    # Improved Interactive Network
    # ------------------------------------------------------------------
    def create_interactive_network(self, save_path=None, max_nodes=600):

        if not PLOTLY_AVAILABLE:
            print("Plotly not installed.")
            return

        if self.graph is None:
            self.load_data_from_pytorch()

        G = self.graph

        # --- Filter important nodes ---
        degrees = dict(G.degree())

        if G.number_of_nodes() > max_nodes:
            print(f"Filtering top {max_nodes} important drugs...")
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()

        print(f"Working with {G.number_of_nodes()} nodes")

        # --- Community Detection ---
        print("Detecting communities...")
        partition = community_louvain.best_partition(G)

        # --- PageRank (better importance measure) ---
        print("Computing PageRank...")
        pagerank = nx.pagerank(G)

        # --- Force layout ---
        print("Computing layout...")
        pos = nx.spring_layout(G, k=1.8, iterations=100, seed=42)

        # -------------------------------
        # Build Plotly traces
        # -------------------------------

        # Edges
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.4, color="rgba(120,120,120,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        )

        # Nodes
        node_x = []
        node_y = []
        node_size = []
        node_color = []
        node_hover = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            importance = pagerank[node]
            degree = degrees[node]

            node_size.append(min(importance * 10000 + 8, 60))
            node_color.append(partition[node])

            name = self.drug_names.get(node, f"Drug_{node}")

            node_text.append("")
            node_hover.append(
                f"<b>{name}</b><br>"
                f"Degree: {degree}<br>"
                f"PageRank: {importance:.4f}<br>"
                f"Community: {partition[node]}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Community"),
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=dict(
                text="<b>Drug Interaction Network (Clustered)</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=24),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=900,
            width=1400,
        )

        if save_path is None:
            save_path = self.data_dir / "interactive_network.html"

        fig.write_html(save_path)
        print(f"\nSaved interactive network to: {save_path}")
        print("Open this file in your browser to explore.")

        return fig

    # ------------------------------------------------------------------
    # Create Static Network Images
    # ------------------------------------------------------------------
    def create_static_network_images(self, max_nodes=400, save_dir=None):
        """
        Create static PNG images of the network showing nodes and connections
        
        Args:
            max_nodes: Maximum number of nodes to visualize
            save_dir: Directory to save images (default: data_dir)
        """
        if self.graph is None:
            self.load_data_from_pytorch()
        
        if save_dir is None:
            save_dir = self.data_dir
        else:
            save_dir = Path(save_dir)
        
        G = self.graph
        
        print(f"\n" + "="*60)
        print("CREATING STATIC NETWORK IMAGES".center(60))
        print("="*60)
        
        # Filter to most connected nodes
        degrees = dict(G.degree())
        if G.number_of_nodes() > max_nodes:
            print(f"Filtering to top {max_nodes} most connected drugs...")
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G_sub = G.subgraph(top_nodes).copy()
        else:
            G_sub = G
        
        print(f"Visualizing {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges():,} edges")
        
        # Community detection for coloring
        try:
            partition = community_louvain.best_partition(G_sub)
            communities = set(partition.values())
            colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
            node_colors = [colors[partition[node]] for node in G_sub.nodes()]
            print(f"Detected {len(communities)} communities")
        except:
            node_colors = 'lightblue'
            print("Using default coloring (install python-louvain for community colors)")
        
        # Node sizes based on degree
        node_sizes = [10 + degrees[node] * 0.5 for node in G_sub.nodes()]
        
        # --- Layout 1: Spring Layout ---
        print("\nGenerating spring layout...")
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        
        pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
        
        # Draw edges (light gray, thin)
        nx.draw_networkx_edges(
            G_sub, pos,
            edge_color='#CCCCCC',
            width=0.3,
            alpha=0.6,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_sub, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.85,
            linewidths=0.5,
            edgecolors='black',
            ax=ax
        )
        
        # Label top nodes
        top_20_percent = int(len(G_sub.nodes()) * 0.15)
        top_nodes_to_label = sorted(G_sub.nodes(), key=lambda n: degrees[n], reverse=True)[:top_20_percent]
        labels = {node: self.drug_names.get(node, f"D{node}") for node in top_nodes_to_label}
        
        nx.draw_networkx_labels(
            G_sub, pos,
            labels=labels,
            font_size=7,
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        ax.set_title(
            f"Drug Interaction Network (Spring Layout)\n{G_sub.number_of_nodes()} Drugs, {G_sub.number_of_edges():,} Interactions",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        ax.margins(0.05)
        
        spring_path = save_dir / "network_spring.png"
        plt.tight_layout()
        plt.savefig(spring_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {spring_path}")
        
        # --- Layout 2: Circular Layout ---
        print("Generating circular layout...")
        fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')
        
        pos = nx.circular_layout(G_sub)
        
        # Draw edges
        nx.draw_networkx_edges(
            G_sub, pos,
            edge_color='#DDDDDD',
            width=0.2,
            alpha=0.4,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_sub, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=0.5,
            edgecolors='black',
            ax=ax
        )
        
        # Label all nodes in circular layout
        all_labels = {node: self.drug_names.get(node, f"D{node}")[:15] for node in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, pos,
            labels=all_labels,
            font_size=6,
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        ax.set_title(
            f"Drug Interaction Network (Circular Layout)\n{G_sub.number_of_nodes()} Drugs, {G_sub.number_of_edges():,} Interactions",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        ax.margins(0.1)
        
        circular_path = save_dir / "network_circular.png"
        plt.tight_layout()
        plt.savefig(circular_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {circular_path}")
        
        # --- Layout 3: Kamada-Kawai Layout (Force-directed) ---
        print("Generating Kamada-Kawai layout...")
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        
        # Use smaller subset for KK layout (it's computationally expensive)
        if G_sub.number_of_nodes() > 200:
            top_kk_nodes = sorted(G_sub.nodes(), key=lambda n: degrees[n], reverse=True)[:200]
            G_kk = G_sub.subgraph(top_kk_nodes).copy()
        else:
            G_kk = G_sub
        
        print(f"  Using {G_kk.number_of_nodes()} nodes for KK layout...")
        pos = nx.kamada_kawai_layout(G_kk)
        
        # Recalculate colors for subset
        try:
            partition_kk = community_louvain.best_partition(G_kk)
            communities_kk = set(partition_kk.values())
            colors_kk = plt.cm.tab20(np.linspace(0, 1, len(communities_kk)))
            node_colors_kk = [colors_kk[partition_kk[node]] for node in G_kk.nodes()]
        except:
            node_colors_kk = 'lightblue'
        
        node_sizes_kk = [15 + degrees[node] * 0.8 for node in G_kk.nodes()]
        
        # Draw edges
        nx.draw_networkx_edges(
            G_kk, pos,
            edge_color='#CCCCCC',
            width=0.4,
            alpha=0.7,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_kk, pos,
            node_color=node_colors_kk,
            node_size=node_sizes_kk,
            alpha=0.9,
            linewidths=0.5,
            edgecolors='black',
            ax=ax
        )
        
        # Label top nodes
        top_kk_labels = sorted(G_kk.nodes(), key=lambda n: degrees[n], reverse=True)[:30]
        labels_kk = {node: self.drug_names.get(node, f"D{node}") for node in top_kk_labels}
        
        nx.draw_networkx_labels(
            G_kk, pos,
            labels=labels_kk,
            font_size=8,
            font_weight='bold',
            font_color='darkred',
            ax=ax
        )
        
        ax.set_title(
            f"Drug Interaction Network (Kamada-Kawai Layout)\nTop {G_kk.number_of_nodes()} Most Connected Drugs, {G_kk.number_of_edges():,} Interactions",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        ax.margins(0.08)
        
        kk_path = save_dir / "network_kamada_kawai.png"
        plt.tight_layout()
        plt.savefig(kk_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {kk_path}")
        
        print("\n" + "="*60)
        print("Static network images created successfully!")
        print("="*60)
        print(f"\n📁 Files saved:")
        print(f"   • {spring_path}")
        print(f"   • {circular_path}")
        print(f"   • {kk_path}")
        print(f"\n💡 Node sizes represent connection degree")
        print(f"💡 Colors represent detected communities")
        print(f"💡 Edges show drug-drug interactions\n")

    # ------------------------------------------------------------------
    # Print Graph Structure
    # ------------------------------------------------------------------
    def print_graph_structure(self):
        """Print detailed information about the graph structure"""
        if self.graph is None:
            print("Graph not loaded yet.")
            return
        
        G = self.graph
        
        print("\n" + "="*70)
        print("DRUG INTERACTION GRAPH STRUCTURE".center(70))
        print("="*70)
        
        # Basic stats
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Drugs (Nodes): {G.number_of_nodes():,}")
        print(f"   Total Interactions (Edges): {G.number_of_edges():,}")
        print(f"   Graph Density: {nx.density(G):.6f}")
        print(f"   Is Connected: {nx.is_connected(G)}")
        
        # Connected components
        components = list(nx.connected_components(G))
        print(f"   Connected Components: {len(components)}")
        if len(components) > 1:
            largest = max(components, key=len)
            print(f"   Largest Component Size: {len(largest):,} nodes ({len(largest)/G.number_of_nodes()*100:.1f}%)")
        
        # Degree statistics
        degrees = dict(G.degree())
        degree_vals = list(degrees.values())
        print(f"\n🔗 DEGREE STATISTICS:")
        print(f"   Average Degree: {np.mean(degree_vals):.2f}")
        print(f"   Median Degree: {np.median(degree_vals):.0f}")
        print(f"   Max Degree: {np.max(degree_vals):,}")
        print(f"   Min Degree: {np.min(degree_vals):,}")
        
        # Top connected drugs
        top_10 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n⭐ TOP 10 MOST CONNECTED DRUGS:")
        print(f"   {'Rank':<6} {'Drug Name':<40} {'Interactions':<12}")
        print("   " + "-"*64)
        for rank, (node_id, degree) in enumerate(top_10, 1):
            drug_name = self.drug_names.get(node_id, f"Drug_{node_id}")
            print(f"   {rank:<6} {drug_name[:38]:<40} {degree:>10,}")
        
        # Community detection if available
        try:
            import community as community_louvain
            print(f"\n🏘️  COMMUNITY DETECTION:")
            partition = community_louvain.best_partition(G)
            num_communities = len(set(partition.values()))
            print(f"   Number of Communities: {num_communities}")
            
            # Community sizes
            from collections import Counter
            comm_sizes = Counter(partition.values())
            sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top 5 Community Sizes:")
            for comm_id, size in sorted_comms:
                print(f"      Community {comm_id}: {size:,} drugs")
                
        except ImportError:
            print(f"\n🏘️  COMMUNITY DETECTION: (install python-louvain for this feature)")
        
        # Clustering coefficient
        print(f"\n📈 NETWORK PROPERTIES:")
        avg_clustering = nx.average_clustering(G)
        print(f"   Average Clustering Coefficient: {avg_clustering:.4f}")
        
        # Diameter (only for connected graphs or largest component)
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"   Graph Diameter: {diameter}")
            print(f"   Average Shortest Path: {avg_path_length:.2f}")
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            print(f"   Diameter (largest component): {diameter}")
            print(f"   Avg Path (largest component): {avg_path_length:.2f}")
        
        print("\n" + "="*70 + "\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():

    print("\n" + "=" * 60)
    print("IMPROVED DRUG INTERACTION VISUALIZATION".center(60))
    print("=" * 60 + "\n")

    viz = DrugNetworkVisualizer()
    viz.load_data_from_pytorch()

    # Create static network images (PNG)
    viz.create_static_network_images(max_nodes=400)
    
    # Create interactive network (HTML)
    viz.create_interactive_network()
    
    # Print detailed graph structure
    viz.print_graph_structure()

    print("\nVisualization complete.\n")


if __name__ == "__main__":
    main()
