"""
Drug Interaction Project - Graph Builder
Converts parsed drug data into graph structure for MT-GAT
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from pathlib import Path
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not found. Chemical features will be skipped.")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

class DrugGraphBuilder:
    """Build graph structure from drug interaction data"""
    
    def __init__(self, drugs_df, interactions_df):
        self.drugs_df = drugs_df
        self.interactions_df = interactions_df
        self.drug_to_idx = None
        self.idx_to_drug = None
        self.graph_data = None
        
    def build_graph(self, use_text_features=True, max_text_features=100):
        """
        Construct graph from drug data
        
        Args:
            use_text_features: Include text embeddings from descriptions
            max_text_features: Max dimensions for text features
        
        Returns:
            PyTorch Geometric Data object
        """
        print("Building drug interaction graph...")
        
        # Create drug ID mappings
        self._create_mappings()
        
        # Build edges (interactions)
        edge_index, edge_attributes = self._build_edges()
        
        # Build node features
        node_features = self._build_node_features(use_text_features, max_text_features)
        
        # Create PyTorch Geometric Data object
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            num_nodes=len(self.drug_to_idx)
        )
        
        print(f"\nâœ… Graph built successfully!")
        print(f"   Nodes (drugs): {self.graph_data.num_nodes}")
        print(f"   Edges (interactions): {self.graph_data.edge_index.shape[1] // 2}")
        print(f"   Node features: {self.graph_data.x.shape[1]}")
        
        return self.graph_data
    
    def _create_mappings(self):
        """Create drug ID to index mappings"""
        unique_drugs = self.drugs_df['drug_id'].unique()
        self.drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(unique_drugs)}
        self.idx_to_drug = {idx: drug_id for drug_id, idx in self.drug_to_idx.items()}
        
        print(f"Created mappings for {len(self.drug_to_idx)} drugs")
    
    def _build_edges(self):
        """Build edge list and attributes from interactions"""
        edge_list = []
        edge_attrs = []
        
        print("Building edges from interactions...")
        for _, row in self.interactions_df.iterrows():
            # Handle different column names
            drug1_id = row.get('drug_1') or row.get('drug_id_1')
            drug2_id = row.get('drug_2') or row.get('drug_id_2')
            description = row.get('description', '')
            
            # Skip if drug not in our drug list
            if drug1_id not in self.drug_to_idx or drug2_id not in self.drug_to_idx:
                continue
            
            drug1_idx = self.drug_to_idx[drug1_id]
            drug2_idx = self.drug_to_idx[drug2_id]
            
            # Add both directions (undirected graph)
            edge_list.append([drug1_idx, drug2_idx])
            edge_list.append([drug2_idx, drug1_idx])
            
            # Edge attributes (severity score)
            severity = self._classify_severity(description)
            edge_attrs.extend([severity, severity])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _classify_severity(self, description):
        """
        Classify interaction severity from description
        Returns: 0.0-1.0 score (higher = more severe)
        """
        if pd.isna(description):
            return 0.5
        
        desc_lower = description.lower()
        
        # Critical keywords
        critical = ['contraindicated', 'avoid', 'do not', 'fatal', 'life-threatening', 
                   'severe', 'serious cardiovascular', 'black box']
        if any(word in desc_lower for word in critical):
            return 1.0
        
        # Major keywords
        major = ['significant', 'substantially', 'major', 'serious', 'marked',
                'requires monitoring', 'dose adjustment', 'toxicity']
        if any(word in desc_lower for word in major):
            return 0.7
        
        # Minor keywords
        minor = ['may increase', 'may decrease', 'slight', 'mild', 'moderate',
                'monitor', 'caution']
        if any(word in desc_lower for word in minor):
            return 0.4
        
        return 0.5  # Default
    
    def _extract_chemical_features(self, smiles):
        """Extract chemical features from SMILES"""
        if not RDKIT_AVAILABLE or not smiles or pd.isna(smiles) or smiles == 'Not Found':
             # Return zeros: 6 descriptors + 128 dimensions for condensed fingerprint (using 128 to keep size manageable)
             # User asked for 2048, but that's huge. I'll use 2048 if I can, but let's stick to descriptors first.
             # Actually user asked for 2048. I will project it or use it. 
             # Let's use 6 descriptors + 2048 bit FP = 2054 features.
             return [0.0] * (6 + 2048)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return [0.0] * (6 + 2048)
                
            # Descriptors
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol)
            ]
            
            # Fingerprint (Morgan, radius 2 = ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_list = [float(b) for b in fp]
            
            return desc + fp_list
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return [0.0] * (6 + 2048)

    
    def _build_node_features(self, use_text_features=True, max_text_features=100):
        """Build feature matrix for all drugs"""
        print("Building node features...")
        
        all_features = []
        
        for drug_id in self.drug_to_idx.keys():
            drug_matches = self.drugs_df[self.drugs_df['drug_id'] == drug_id]
            if len(drug_matches) == 0:
                # Drug not in our dataset, use default features
                features = self._get_default_features()
            else:
                drug_data = drug_matches.iloc[0]
                features = self._extract_drug_features(drug_data)
            all_features.append(features)
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Add text features if requested
        if use_text_features:
            text_features = self._extract_text_features(max_text_features)
            features_array = np.hstack([features_array, text_features])
        
        # Normalize features
        scaler = StandardScaler()
        features_array = scaler.fit_transform(features_array)
        
        # Convert to tensor
        node_features = torch.tensor(features_array, dtype=torch.float)
        
        return node_features
    
    def _extract_drug_features(self, drug_data):
        """Extract numerical features for a single drug"""
        features = []
        
        # Drug type (one-hot)
        features.append(1.0 if drug_data['drug_type'] == 'small_molecule' else 0.0)
        features.append(1.0 if drug_data['drug_type'] == 'biotech' else 0.0)
        
        # State
        features.append(1.0 if drug_data['state'] == 'solid' else 0.0)
        features.append(1.0 if drug_data['state'] == 'liquid' else 0.0)
        
        # Groups (approved, experimental, etc.)
        groups = str(drug_data.get('groups', ''))
        features.append(1.0 if 'approved' in groups else 0.0)
        features.append(1.0 if 'experimental' in groups else 0.0)
        features.append(1.0 if 'withdrawn' in groups else 0.0)
        
        # Text length features (proxy for information richness)
        features.append(len(str(drug_data.get('description', ''))) / 1000.0)
        features.append(len(str(drug_data.get('indication', ''))) / 1000.0)
        
        # Category features (most common categories)
        categories = str(drug_data.get('categories', ''))
        common_categories = [
            'Anticoagulants', 'Antibiotics', 'Analgesics', 'Antidepressants',
            'Antihypertensive', 'Anti-inflammatory', 'Cardiovascular', 'CNS'
        ]
        for cat in common_categories:
            features.append(1.0 if cat.lower() in categories.lower() else 0.0)
        
        # Chemical features
        if 'smiles' in drug_data:
            chem_features = self._extract_chemical_features(drug_data['smiles'])
            features.extend(chem_features)
            
        return features
    
    def _get_default_features(self):
        """Return default features for drugs not in dataset"""
        # Match the feature count from _extract_drug_features
        num_basic_features = 2 + 2 + 3 + 2  # type + state + groups + text_length
        num_categories = 8  # common categories
        
        # Add chemical features count (6 descriptors + 2048 fingerprint)
        # Only if we are using them. For now we assume we are if the column exists in DF.
        # But we need to be consistent. 
        # If the input DF has 'smiles', _extract_drug_features will add 2054 cols.
        # So default features must match.
        
        num_chemical = 6 + 2048
        total_features = num_basic_features + num_categories + num_chemical
        return [0.0] * total_features
    
    def _extract_text_features(self, max_features=100):
        """Extract TF-IDF features from drug descriptions"""
        print("  Extracting text features using TF-IDF...")
        
        # Collect all descriptions
        descriptions = []
        for drug_id in self.drug_to_idx.keys():
            drug_matches = self.drugs_df[self.drugs_df['drug_id'] == drug_id]
            if len(drug_matches) == 0:
                descriptions.append('unknown')
            else:
                drug_data = drug_matches.iloc[0]
                desc = str(drug_data.get('description', ''))
                mech = str(drug_data.get('mechanism_of_action', ''))
                text = desc + ' ' + mech
                descriptions.append(text if text.strip() else 'unknown')
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        text_features = vectorizer.fit_transform(descriptions).toarray()
        
        return text_features
    
    def save_graph(self, filepath='data/drug_graph.pt'):
        """Save graph and mappings"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save graph
        torch.save({
            'graph_data': self.graph_data,
            'drug_to_idx': self.drug_to_idx,
            'idx_to_drug': self.idx_to_drug
        }, filepath)
        
        print(f"\nâœ… Graph saved to {filepath}")
    
    @staticmethod
    def load_graph(filepath='data/drug_graph.pt'):
        """Load saved graph"""
        data = torch.load(filepath)
        return data['graph_data'], data['drug_to_idx'], data['idx_to_drug']
    
    def visualize_graph(self, output_dir='data', max_nodes=300, dpi=150):
        """
        Create visualization images of the graph structure
        
        Args:
            output_dir: Directory to save images
            max_nodes: Maximum nodes to visualize (for performance)
            dpi: Resolution of output images
        """
        if self.graph_data is None or self.idx_to_drug is None:
            print("âŒ Graph not built yet. Call build_graph() first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("VISUALIZING GRAPH STRUCTURE".center(70))
        print("="*70)
        
        # Convert to NetworkX graph for visualization
        print("\nConverting to NetworkX format...")
        G = nx.Graph()
        
        # Add nodes
        num_nodes = self.graph_data.x.shape[0]
        for i in range(num_nodes):
            drug_name = self.idx_to_drug.get(i, f"Drug_{i}")
            G.add_node(i, label=drug_name)
        
        # Add edges
        edge_index = self.graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src != dst:  # Skip self-loops
                G.add_edge(src, dst)
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Filter to most connected nodes if needed
        if G.number_of_nodes() > max_nodes:
            print(f"Filtering to top {max_nodes} connected nodes...")
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()
            print(f"Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Get node properties
        degrees = dict(G.degree())
        node_sizes = [10 + degrees[node] * 0.3 for node in G.nodes()]
        
        # Community detection for colors
        try:
            if LOUVAIN_AVAILABLE:
                print("Detecting communities...")
                partition = community_louvain.best_partition(G)
                communities = set(partition.values())
                colors = plt.cm.tab20(np.linspace(0, 1, min(len(communities), 20)))
                node_colors = [colors[partition[node] % 20] for node in G.nodes()]
                num_communities = len(communities)
                print(f"Found {num_communities} communities")
            else:
                node_colors = 'lightblue'
                print("Using default coloring (install python-louvain for communities)")
        except:
            node_colors = 'lightblue'
            print("Using default coloring")
        
        # --- Layout 1: Spring Layout ---
        print("\nGenerating spring layout...")
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='#0E1117')
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw edges (subtle)
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#888888',
            width=0.3,
            alpha=0.15,
            ax=ax
        )
        
        # Draw nodes with white borders
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=1.0,
            edgecolors='white',
            ax=ax
        )
        
        # Label top nodes in white
        top_labels_count = max(10, len(G.nodes()) // 20)
        top_nodes_to_label = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)[:top_labels_count]
        labels = {node: self.idx_to_drug.get(node, f"D{node}")[:12] for node in top_nodes_to_label}
        
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=7,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        ax.set_title(
            f"Drug Interaction Network (Spring Layout)\n{G.number_of_nodes()} Drugs, {G.number_of_edges():,} Interactions",
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=15
        )
        ax.axis('off')
        ax.margins(0.05)
        
        spring_path = output_path / "graph_spring_layout.png"
        plt.tight_layout()
        plt.savefig(spring_path, dpi=dpi, bbox_inches='tight', facecolor='#0E1117')
        plt.close()
        print(f"âœ… Saved: {spring_path}")
        
        # --- Layout 2: Circular Layout ---
        print("Generating circular layout...")
        fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0E1117')
        
        pos = nx.circular_layout(G)
        
        # Draw edges (subtle)
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#888888',
            width=0.3,
            alpha=0.12,
            ax=ax
        )
        
        # Draw nodes with white borders
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=1.0,
            edgecolors='white',
            ax=ax
        )
        
        # Label all nodes in white
        all_labels = {node: self.idx_to_drug.get(node, f"D{node}")[:10] for node in list(G.nodes())[:50]}  # Label first 50
        nx.draw_networkx_labels(
            G, pos,
            labels=all_labels,
            font_size=5,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        ax.set_title(
            f"Drug Interaction Network (Circular Layout)\n{G.number_of_nodes()} Drugs, {G.number_of_edges():,} Interactions",
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=15
        )
        ax.axis('off')
        ax.margins(0.1)
        
        circular_path = output_path / "graph_circular_layout.png"
        plt.tight_layout()
        plt.savefig(circular_path, dpi=dpi, bbox_inches='tight', facecolor='#0E1117')
        plt.close()
        print(f"âœ… Saved: {circular_path}")
        
        # --- Layout 3: Degree Distribution ---
        print("Generating degree distribution...")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        degree_values = list(degrees.values())
        ax.hist(degree_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Node Degree (Number of Interactions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(
            f"Degree Distribution of Drug Interaction Network\nMean: {np.mean(degree_values):.2f}, Max: {np.max(degree_values)}",
            fontsize=12,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        dist_path = output_path / "graph_degree_distribution.png"
        plt.tight_layout()
        plt.savefig(dist_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"âœ… Saved: {dist_path}")
        
        # Summary statistics
        print("\n" + "="*70)
        print("GRAPH VISUALIZATION COMPLETE".center(70))
        print("="*70)
        print(f"\nðŸ“Š Network Statistics:")
        print(f"   Nodes: {G.number_of_nodes():,}")
        print(f"   Edges: {G.number_of_edges():,}")
        print(f"   Average Degree: {np.mean(degree_values):.2f}")
        print(f"   Max Degree: {np.max(degree_values):,}")
        print(f"   Min Degree: {np.min(degree_values)}")
        print(f"   Density: {nx.density(G):.6f}")
        
        print(f"\nðŸ“ Images saved to {output_path}:")
        print(f"   â€¢ graph_spring_layout.png")
        print(f"   â€¢ graph_circular_layout.png")
        print(f"   â€¢ graph_degree_distribution.png\n")
        
        return G
    
    def create_3d_interactive_visualization(self, graph=None, output_dir='data', max_nodes=300):

        if not PLOTLY_AVAILABLE:
            print("âš ï¸  Plotly not installed. Install with: pip install plotly")
            return

        if self.graph_data is None or self.idx_to_drug is None:
            print("âŒ Graph not built yet. Call build_graph() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("CREATING CLEAN 3D INTERACTIVE VISUALIZATION".center(70))
        print("="*70)

        # ---------------- BUILD GRAPH ----------------
        if graph is None:
            G = nx.Graph()
            num_nodes = self.graph_data.x.shape[0]

            for i in range(num_nodes):
                G.add_node(i)

            edge_index = self.graph_data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src != dst:
                    G.add_edge(src, dst)
        else:
            G = graph

        degrees = dict(G.degree())

        # Filter top connected nodes
        if G.number_of_nodes() > max_nodes:
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()

        degrees = dict(G.degree())
        max_degree = max(degrees.values())

        print(f"Visualizing {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")

        # ---------------- TRUE 3D SPRING LAYOUT ----------------
        pos = nx.spring_layout(G, k=1.5, iterations=40, seed=42, dim=3)

        # ---------------- EDGE TRACE ----------------
        edge_x, edge_y, edge_z = [], [], []

        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(200,200,200,0.05)',  # subtle edges
                width=0.5
            ),
            hoverinfo='none',
            showlegend=False
        )

        # ---------------- NODE TRACE ----------------
        node_x, node_y, node_z = [], [], []
        node_sizes, node_colors, node_labels = [], [], []

        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            deg = degrees[node]
            node_sizes.append(6 + (deg / max_degree) * 14)
            node_colors.append(deg)
            node_labels.append(self.idx_to_drug.get(node, f"Drug_{node}"))

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Node Degree",
                    thickness=15,
                    len=0.6,
                    tickfont=dict(color='white'),
                    bgcolor='#0E1117'
                ),
                line=dict(color='white', width=1),
                opacity=0.95
            ),
            hovertext=node_labels,
            hoverinfo='text',
            showlegend=False
        )

        # ---------------- FIGURE ----------------
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title={
                'text': f"<b style='color:white'>3D Drug Interaction Network</b><br>"
                        f"<span style='color:rgba(255,255,255,0.7)'>"
                        f"{G.number_of_nodes()} Drugs, {G.number_of_edges():,} Interactions</span>",
                'x': 0.5
            },
            scene=dict(
                xaxis=dict(showgrid=False, showbackground=False, visible=False),
                yaxis=dict(showgrid=False, showbackground=False, visible=False),
                zaxis=dict(showgrid=False, showbackground=False, visible=False),
                bgcolor='#0E1117',
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.6))
            ),
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            margin=dict(l=0, r=0, t=80, b=0),
            width=1400,
            height=900,
            font=dict(color='white')
        )

        html_path = output_path / "graph_3d_interactive_clean.html"
        fig.write_html(html_path)

        print(f"\nâœ… Clean 3D visualization saved: {html_path}")
        print("   â€¢ Rotate: Click + Drag")
        print("   â€¢ Zoom: Scroll")
        print("   â€¢ Hover nodes to see drug names\n")

        return fig
    
    def get_statistics(self):
        """Get graph statistics"""
        if self.graph_data is None:
            return None
        
        num_edges = self.graph_data.edge_index.shape[1] // 2  # Divide by 2 for undirected
        avg_degree = num_edges * 2 / self.graph_data.num_nodes
        
        stats = {
            'num_nodes': self.graph_data.num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'num_features': self.graph_data.x.shape[1],
            'density': num_edges / (self.graph_data.num_nodes * (self.graph_data.num_nodes - 1) / 2)
        }
        
        return stats


def main():
    """Example usage"""
    
    # Load parsed data
    print("Loading data...")
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    print(f"Loaded {len(drugs_df)} drugs and {len(interactions_df)} interactions")
    
    # Build graph
    builder = DrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(use_text_features=True, max_text_features=50)
    
    # Show statistics
    stats = builder.get_statistics()
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Visualize graph
    builder.visualize_graph(output_dir='data', max_nodes=300)
    
    # Create 3D interactive visualization
    builder.create_3d_interactive_visualization(output_dir='data', max_nodes=300)
    
    # Save graph
    builder.save_graph('data/drug_graph.pt')
    
    print("\nâœ… Graph construction and visualization complete!")
    print("   Ready for MT-GAT training.")


if __name__ == "__main__":
    main()

