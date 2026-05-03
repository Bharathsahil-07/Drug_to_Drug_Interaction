"""
Drug Interaction Project - Rigorous Graph Builder
Avoids data leakage by decoupling feature fitting from transformation.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class RigorousDrugGraphBuilder:
    def __init__(self, drugs_df, interactions_df):
        self.drugs_df = drugs_df
        self.interactions_df = interactions_df
        self.drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drugs_df['drug_id'].unique())}
        self.idx_to_drug = {idx: drug_id for drug_id, idx in self.drug_to_idx.items()}
        
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_fitted = False

    def get_raw_features(self):
        """Extract raw numerical and text features without scaling or TF-IDF fitting"""
        numerical_features = []
        descriptions = []
        
        for drug_id in self.drug_to_idx.keys():
            drug_data = self.drugs_df[self.drugs_df['drug_id'] == drug_id].iloc[0]
            
            # 1. Basic & Pharmacological Features
            feat = []
            feat.append(1.0 if drug_data['drug_type'] == 'small_molecule' else 0.0)
            feat.append(1.0 if drug_data['drug_type'] == 'biotech' else 0.0)
            feat.append(1.0 if drug_data['state'] == 'solid' else 0.0)
            feat.append(1.0 if drug_data['state'] == 'liquid' else 0.0)
            
            groups = str(drug_data.get('groups', ''))
            feat.append(1.0 if 'approved' in groups else 0.0)
            feat.append(1.0 if 'experimental' in groups else 0.0)
            feat.append(1.0 if 'withdrawn' in groups else 0.0)
            
            feat.append(len(str(drug_data.get('description', ''))) / 1000.0)
            feat.append(len(str(drug_data.get('indication', ''))) / 1000.0)
            
            common_categories = [
                'Anticoagulants', 'Antibiotics', 'Analgesics', 'Antidepressants',
                'Antihypertensive', 'Anti-inflammatory', 'Cardiovascular', 'CNS'
            ]
            categories = str(drug_data.get('categories', ''))
            for cat in common_categories:
                feat.append(1.0 if cat.lower() in categories.lower() else 0.0)
            
            # 2. Chemical Features (SMILES)
            if 'smiles' in drug_data and RDKIT_AVAILABLE:
                smiles = drug_data['smiles']
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        feat.extend([
                            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)
                        ])
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        feat.extend([float(b) for b in fp])
                    else:
                        feat.extend([0.0] * (6 + 2048))
                except:
                    feat.extend([0.0] * (6 + 2048))
            else:
                feat.extend([0.0] * (6 + 2048))
            
            numerical_features.append(feat)
            
            # 3. Text for TF-IDF
            desc = str(drug_data.get('description', ''))
            mech = str(drug_data.get('mechanism_of_action', ''))
            descriptions.append(desc + " " + mech)
            
        return np.array(numerical_features), descriptions

    def build_graph(self, train_indices):
        """Build graph where scaler and TF-IDF are fitted ONLY on train_indices"""
        num_feats, texts = self.get_raw_features()
        
        # Fit on training set only (Fixes Leakage)
        self.scaler.fit(num_feats[train_indices])
        self.vectorizer.fit([texts[i] for i in train_indices])
        
        # Transform all
        num_feats_scaled = self.scaler.transform(num_feats)
        text_feats = self.vectorizer.transform(texts).toarray()
        
        combined_features = np.hstack([num_feats_scaled, text_feats])
        x = torch.tensor(combined_features, dtype=torch.float)
        
        # Build Edges
        edge_list = []
        edge_attrs = []
        
        for _, row in self.interactions_df.iterrows():
            d1, d2 = row['drug_1'], row['drug_2']
            if d1 in self.drug_to_idx and d2 in self.drug_to_idx:
                u, v = self.drug_to_idx[d1], self.drug_to_idx[d2]
                edge_list.extend([[u, v], [v, u]])
                
                # Severity mapping
                desc = str(row.get('description', '')).lower()
                sev = 0.5
                if any(w in desc for w in ['contraindicated', 'avoid', 'fatal', 'life-threatening']): sev = 1.0
                elif any(w in desc for w in ['significant', 'major', 'serious', 'toxicity']): sev = 0.7
                elif any(w in desc for w in ['minor', 'mild', 'moderate', 'caution']): sev = 0.4
                edge_attrs.extend([sev, sev])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(self.drug_to_idx))

    def get_severity_weights(self, edge_index, edge_attr):
        """Calculate class weights for severity loss to fix imbalance"""
        # Mapping: 0.4 -> 0, 0.7 -> 1, 1.0 -> 2
        labels = []
        for val in edge_attr.squeeze().tolist():
            if val <= 0.5: labels.append(0)
            elif val <= 0.8: labels.append(1)
            else: labels.append(2)
        
        counts = np.bincount(labels, minlength=3)
        total = len(labels)
        weights = total / (3 * (counts + 1e-6))
        return torch.tensor(weights, dtype=torch.float)
