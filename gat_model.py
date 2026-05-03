

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

class DrugInteractionMTGAT(nn.Module):
    """
    Multi-Task GAT for Drug Interaction Prediction
    Heads: Interaction (Binary), Severity (Multi-class), Confidence (Reg)
    """
    
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.2, heads=4):
        super(DrugInteractionMTGAT, self).__init__()
        
        # GAT layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, embedding_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # Interaction Decoder (Binary)
        self.interaction_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Severity Head (3 classes: Minor, Moderate, Major)
        self.severity_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3) # Classes: 0=Minor, 1=Moderate, 2=Major
        )
        
        # Confidence Head (Regression 0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x, edge_index, return_attention=False):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 3
        if return_attention:
            # We get attention weights from the last layer
            # GATConv returns (x, (edge_index, attention_weights))
            x, att_data = self.conv3(x, edge_index, return_attention_weights=True)
            return x, att_data
        
        x = self.conv3(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """
        Predict interaction probability for drug pairs
        
        Args:
            z: Drug embeddings [num_nodes, embedding_dim]
            edge_index: Pairs to predict [2, num_pairs]
        
        Returns:
            Interaction probabilities [num_pairs, 1]
        """
        # Get embeddings for each drug in pair
        drug1_embed = z[edge_index[0]]
        drug2_embed = z[edge_index[1]]
        
        # Concatenate pair embeddings
        pair_embed = torch.cat([drug1_embed, drug2_embed], dim=1)
        
        # Heads
        interaction_logits = self.interaction_decoder(pair_embed)
        severity_logits = self.severity_head(pair_embed)
        confidence = self.confidence_head(pair_embed)
        
        return interaction_logits, severity_logits, confidence
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Full forward pass
        
        Args:
            x: Node features
            edge_index: Training edges (message passing)
            edge_label_index: Edges to predict
        
        Returns:
            Predictions for edge_label_index
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
    def predict_interaction(self, x, edge_index, drug1_idx, drug2_idx, return_attention=False):
        """
        Predict interaction between two specific drugs
        
        Args:
            x: Node features
            edge_index: Graph edges
            drug1_idx: Index of first drug
            drug2_idx: Index of second drug
        
        Returns:
            Interaction probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            if return_attention:
                z, (att_edge_index, att_weights) = self.encode(x, edge_index, return_attention=True)
            else:
                z = self.encode(x, edge_index)
                
            test_edge = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long).to(x.device)
            int_logit, sev_logit, conf = self.decode(z, test_edge)
            
            prob = torch.sigmoid(int_logit).item()
            severity_probs = torch.softmax(sev_logit, dim=1).squeeze().tolist() # [Minor, Moderate, Major]
            confidence = conf.item()
            
            # Map severity index to label
            severity_labels = ['Minor', 'Moderate', 'Major']
            severity_idx = torch.argmax(sev_logit, dim=1).item()
            severity = severity_labels[severity_idx]
            
            if return_attention:
                return prob, severity, confidence, severity_probs, (att_edge_index, att_weights)
            
        return prob, severity, confidence, severity_probs


class MTGATTrainer:
    """Trainer for Drug Interaction MT-GAT"""
    
    def __init__(self, model, graph_data, device='cpu'):
        self.model = model.to(device)
        self.graph_data = graph_data.to(device)
        self.device = device
        self.train_losses = []
        self.val_aucs = []
        
        # Multi-task labels
        self.train_severity_labels = None
        self.val_severity_labels = None
        self.test_severity_labels = None
    
    def split_edges(self, train_ratio=0.8, val_ratio=0.1):
        """
        Split edges into train/val/test sets
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        edge_index = self.graph_data.edge_index
        num_edges = edge_index.shape[1] // 2  # Undirected, so divide by 2
        
        # Get unique edges (remove reverse edges)
        edge_list = edge_index.t().tolist()
        unique_edges = []
        seen = set()
        for e in edge_list:
            edge_tuple = tuple(sorted(e))
            if edge_tuple not in seen:
                unique_edges.append(e)
                seen.add(edge_tuple)
        
        unique_edges = torch.tensor(unique_edges).t()
        
        # Shuffle
        perm = torch.randperm(unique_edges.shape[1])
        unique_edges = unique_edges[:, perm]
        
        # Split
        num_train = int(train_ratio * unique_edges.shape[1])
        num_val = int(val_ratio * unique_edges.shape[1])
        
        self.train_edges = unique_edges[:, :num_train]
        self.val_edges = unique_edges[:, num_train:num_train+num_val]
        self.test_edges = unique_edges[:, num_train+num_val:]
        
        # Store severity labels for heads (mapped from edge_attr)
        # Mapping: 0.4 -> 0 (Minor), 0.7 -> 1 (Moderate), 1.0 -> 2 (Major)
        def map_severity(val):
            if val <= 0.5: return 0
            if val <= 0.8: return 1
            return 2
            
        # Get labels from edge_attr for the unique edges
        # Optimization: use a dictionary to map (u,v) -> severity
        edge_attr = self.graph_data.edge_attr
        edge_to_sev = {}
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (u, v) not in edge_to_sev:
                edge_to_sev[(u, v)] = edge_attr[i].item()
                
        all_unique_labels = []
        for i in range(unique_edges.shape[1]):
            u, v = unique_edges[0, i].item(), unique_edges[1, i].item()
            val = edge_to_sev.get((u, v), 0.5)
            all_unique_labels.append(map_severity(val))
            
        all_unique_labels = torch.tensor(all_unique_labels, dtype=torch.long)
        self.train_severity_labels = all_unique_labels[:num_train]
        self.val_severity_labels = all_unique_labels[num_train:num_train+num_val]
        self.test_severity_labels = all_unique_labels[num_train+num_val:]
        
        # Add reverse edges for message passing
        self.train_edges_undirected = torch.cat([
            self.train_edges,
            torch.stack([self.train_edges[1], self.train_edges[0]])
        ], dim=1)
        
        print(f"Edge split:")
        print(f"  Train: {self.train_edges.shape[1]} edges")
        print(f"  Val: {self.val_edges.shape[1]} edges")
        print(f"  Test: {self.test_edges.shape[1]} edges")
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Positive samples
        pos_edge_index = self.train_edges
        
        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=self.train_edges_undirected,
            num_nodes=self.graph_data.num_nodes,
            num_neg_samples=pos_edge_index.shape[1]
        )
        
        # Combine positive and negative
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([
            torch.ones(pos_edge_index.shape[1], 1),
            torch.zeros(neg_edge_index.shape[1], 1)
        ], dim=0).to(self.device)
        
        # Forward pass
        logits, sev_logits, conf = self.model(
            self.graph_data.x,
            self.train_edges_undirected,
            edge_label_index
        )
        
        # -- Loss Calculation --
        # 1. Interaction Loss (BCE)
        loss_interaction = criterion(logits, edge_labels)
        
        # 2. Severity Loss (CrossEntropy) - Only for positive edges
        loss_severity = torch.tensor(0.0).to(self.device)
        if pos_edge_index.shape[1] > 0:
            # Mask to get only positive logits for severity
            pos_sev_logits = sev_logits[:pos_edge_index.shape[1]]
            
            # Extract ground truth severity from edge_attr
            # We need to map pos_edge_index back to original edge_attr indices
            # Since train_edges was sliced from unique_edges, we need to track indices or just use edge_attr directly if aligned.
            # In split_edges, we shuffled unique_edges. Let's ensure we have the labels.
            
            if hasattr(self, 'train_severity_labels'):
                target_severity = self.train_severity_labels.to(self.device)
                loss_severity = F.cross_entropy(pos_sev_logits, target_severity)
        
        # 3. Confidence Loss - Regress towards 1.0 for known interactions
        loss_confidence = F.mse_loss(conf, edge_labels)
        
        # Total
        loss = loss_interaction + 0.5 * loss_severity + 0.2 * loss_confidence
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, edge_index):
        """Evaluate on given edge set"""
        self.model.eval()
        
        with torch.no_grad():
            # Positive samples
            pos_edge_index = edge_index
            
            # Negative samples
            neg_edge_index = negative_sampling(
                edge_index=self.train_edges_undirected,
                num_nodes=self.graph_data.num_nodes,
                num_neg_samples=pos_edge_index.shape[1]
            )
            
            # Combine
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.shape[1]),
                torch.zeros(neg_edge_index.shape[1])
            ]).cpu().numpy()
            
            # Predict
            logits, sev_logits, conf = self.model(
                self.graph_data.x,
                self.train_edges_undirected,
                edge_label_index
            )
            
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Severity Predictions
            sev_probs = torch.softmax(sev_logits, dim=1).cpu().numpy()
            sev_preds = np.argmax(sev_probs, axis=1)
            
            # Metrics
            if len(np.unique(edge_labels)) > 1:
                auc = roc_auc_score(edge_labels, probs)
            else:
                auc = 0.5
                
            acc = accuracy_score(edge_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                edge_labels, preds, average='binary', zero_division=0
            )
            
            # Severity Accuracy (only for true positives)
            sev_acc = 0.0
            if hasattr(self, 'val_severity_labels') and self.val_severity_labels is not None:
                # This is tricky because edge_label_index has negatives too.
                # We only want to evaluate severity on the original positive edges.
                target_labels = None
                if torch.equal(edge_index, self.val_edges):
                    target_labels = self.val_severity_labels
                elif torch.equal(edge_index, self.test_edges):
                    target_labels = self.test_severity_labels
                
                if target_labels is not None:
                    # Positive edges are the first half
                    num_pos = len(target_labels)
                    pos_sev_preds = sev_preds[:num_pos]
                    sev_acc = accuracy_score(target_labels.cpu().numpy(), pos_sev_preds)
        
        return {
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'severity_acc': sev_acc
        }
    
    def train(self, epochs=100, lr=0.001, weight_decay=5e-4):
        """
        Full training loop
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        patience = 20
        patience_counter = 0
        
        print("\n" + "="*70)
        print("TRAINING PROGRESS".center(70))
        print("="*70)
        print(f"{'Epoch':>6} {'Loss':>10} {'Val AUC':>10} {'Val Acc':>10} {'Val F1':>10} {'Sev Acc':>10} {'Status':>15}")
        print("="*85)
        
        for epoch in range(epochs):
            # Train
            loss = self.train_epoch(optimizer, criterion)
            self.train_losses.append(loss)
            
            # Validate every 1 epoch
            if (epoch + 1) % 1 == 0:
                val_metrics = self.evaluate(self.val_edges)
                self.val_aucs.append(val_metrics['auc'])
                
                status = "[BEST]" if val_metrics['auc'] > best_val_auc else ""
                print(f"{epoch+1:>6} {loss:>10.4f} {val_metrics['auc']:>10.4f} {val_metrics['accuracy']:>10.4f} {val_metrics['f1']:>10.4f} {val_metrics['severity_acc']:>10.4f} {status:>15}")
                
                # Early stopping
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'data/best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("="*70)
                    print(f"\n[STOP] Early stopping at epoch {epoch+1} (no improvement for {patience} checks)")
                    break
            elif (epoch + 1) % 1 == 0:
                # Print loss every epoch (without validation)
                print(f"{epoch+1:>6} {loss:>10.4f} {'---':>10} {'---':>10} {'---':>10} {'Training':>15}")
        
        # Load best model
        self.model.load_state_dict(torch.load('data/best_model.pt'))
        
        # Final test evaluation
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        test_metrics = self.evaluate(self.test_edges)
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return test_metrics
    
    def plot_training(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(self.train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # AUC curve
        ax2.plot(self.val_aucs)
        ax2.set_xlabel('Validation Step')
        ax2.set_ylabel('AUC')
        ax2.set_title('Validation AUC')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/training_curves.png', dpi=150)
        print("Training curves saved to data/training_curves.png")


def main():
    """Example usage"""
    from graph_builder import DrugGraphBuilder
    
    # Load graph
    print("Loading graph...")
    graph_data, drug_to_idx, idx_to_drug = DrugGraphBuilder.load_graph('data/drug_graph.pt')
    
    # Initialize model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=256, embedding_dim=128, heads=4)
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: 256")
    print(f"  Embedding dim: 128")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = MTGATTrainer(model, graph_data, device=device)
    trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
    
    test_metrics = trainer.train(epochs=200, lr=0.001)
    
    # Plot
    trainer.plot_training()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug_to_idx': drug_to_idx,
        'idx_to_drug': idx_to_drug,
        'input_dim': input_dim
    }, 'data/trained_model.pt')
    
    print("\nâœ… Training complete! Model saved to data/trained_model.pt")


if __name__ == "__main__":
    main()

