import numpy as np
import pandas as pd
import torch
import re

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class FeatureImportance:
    def __init__(self, drugs_df, model=None):
        self.drugs_df = drugs_df
        self.model = model
        self.drug_meta = self.drugs_df.set_index('drug_id').to_dict('index')

    def compute_contributions(self, drug1_id, drug2_id, embeddings=None, drug_to_idx=None):
        """Compute similarity scores across multiple feature spaces"""
        d1 = self.drug_meta.get(drug1_id, {})
        d2 = self.drug_meta.get(drug2_id, {})
        
        scores = {
            "chemical_similarity": 0.0,
            "target_overlap": 0.0,
            "atc_similarity": 0.0,
            "graph_context": 0.0
        }
        
        if not d1 or not d2:
            return scores
            
        # 1. Chemical Similarity (Tanimoto on Morgan FP)
        if RDKIT_AVAILABLE:
            s1 = d1.get('smiles')
            s2 = d2.get('smiles')
            if s1 and s2 and s1 != 'Not Found' and s2 != 'Not Found':
                try:
                    m1 = Chem.MolFromSmiles(s1)
                    m2 = Chem.MolFromSmiles(s2)
                    if m1 and m2:
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
                        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
                        scores["chemical_similarity"] = DataStructs.TanimotoSimilarity(fp1, fp2)
                except:
                    pass
        
        # 2. ATC Similarity (Prefix matching)
        atc1 = d1.get('atc_code')
        atc2 = d2.get('atc_code')
        if atc1 and atc2 and atc1 != 'Not Found' and atc2 != 'Not Found':
            # Compare prefixes: Level 1 (1 char), Level 2 (3 chars), Level 3 (4 chars), Level 4 (5 chars)
            matches = 0
            for length in [1, 3, 4, 5]:
                if atc1[:length] == atc2[:length]:
                    matches += 0.25
            scores["atc_similarity"] = matches

        # 3. Target Overlap (Token-based proxy on mechanism)
        mech1 = str(d1.get('mechanism_of_action', '')).lower()
        mech2 = str(d2.get('mechanism_of_action', '')).lower()
        if mech1 and mech2:
            # Extract potential targets (words that look like proteins/receptors)
            tokens1 = set(re.findall(r'\b[a-z]{2,8}\d?[a-z]?\b', mech1))
            tokens2 = set(re.findall(r'\b[a-z]{2,8}\d?[a-z]?\b', mech2))
            # Filter common stop words or non-target tokens (simplified)
            common = tokens1.intersection(tokens2)
            if tokens1 or tokens2:
                scores["target_overlap"] = len(common) / max(len(tokens1), len(tokens2), 1)

        # 4. Graph Context (Cosine similarity on embeddings)
        if embeddings is not None and drug_to_idx is not None:
            idx1 = drug_to_idx.get(drug1_id)
            idx2 = drug_to_idx.get(drug2_id)
            if idx1 is not None and idx2 is not None:
                v1 = embeddings[idx1]
                v2 = embeddings[idx2]
                cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
                scores["graph_context"] = float(cos.item())

        # Normalize and round
        return {k: round(float(v), 3) for k, v in scores.items()}
