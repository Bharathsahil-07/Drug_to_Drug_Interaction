import torch
import numpy as np

class ModelExplainer:
    def __init__(self, model, graph_data, idx_to_drug_name):
        self.model = model
        self.graph_data = graph_data
        self.idx_to_drug_name = idx_to_drug_name

    def explain_prediction(self, drug1_idx, drug2_idx):
        """
        Extract GAT attention weights and identify top influencing neighbors.
        Part 2 & 6 Implementation
        """
        self.model.eval()
        with torch.no_grad():
            z, (edge_index, attention) = self.model.encode(
                self.graph_data.x, 
                self.graph_data.edge_index, 
                return_attention=True
            )
            
            avg_attention = attention.mean(dim=1).cpu().numpy()
            edge_index_np = edge_index.cpu().numpy()

            # Fix Part 2: Normalize relative to max weight
            max_attention = float(np.max(avg_attention)) if len(avg_attention) > 0 else 0.001
            mean_attention = float(np.mean(avg_attention)) if len(avg_attention) > 0 else 0.001
            
            # Find influential neighbors
            neighbors1 = self._get_top_neighbors(drug1_idx, edge_index_np, avg_attention, max_attention)
            neighbors2 = self._get_top_neighbors(drug2_idx, edge_index_np, avg_attention, max_attention)

            # Combine and sort
            all_neighbors = neighbors1 + neighbors2
            all_neighbors.sort(key=lambda x: x['relative_influence'], reverse=True)

            # Part 6: Dynamic reasoning text
            reasoning = "Prediction derived from aggregated graph context across multiple neighboring drugs."
            if max_attention > 5 * mean_attention:
                reasoning = "Prediction strongly influenced by graph proximity to drugs with highly similar clinical profiles."
            elif max_attention < 1.5 * mean_attention:
                reasoning = "Prediction influenced by distributed graph context rather than a single dominant neighbor."

            return {
                "top_influencing_neighbors": all_neighbors[:6],
                "attention_score": round(max_attention, 4),
                "confidence_reasoning": reasoning,
                "is_flat_attention": max_attention < 1.5 * mean_attention
            }

    def _get_top_neighbors(self, node_idx, edge_index, attention, max_val, top_k=3):
        """Find the most important neighbors for a specific node"""
        mask = edge_index[1] == node_idx
        relevant_indices = np.where(mask)[0]
        
        neighbors = []
        for idx in relevant_indices:
            neighbor_idx = int(edge_index[0, idx])
            if neighbor_idx == node_idx: continue 
            
            weight = float(attention[idx])
            # Normalize: (weight / max_val) * 100
            rel_influence = (weight / max_val) * 100 if max_val > 0 else 0
            
            neighbors.append({
                "drug": self.idx_to_drug_name.get(neighbor_idx, f"Drug_{neighbor_idx}"),
                "attention_weight": round(weight, 5),
                "relative_influence": round(rel_influence, 1)
            })
        
        neighbors.sort(key=lambda x: x['relative_influence'], reverse=True)
        return neighbors[:top_k]
