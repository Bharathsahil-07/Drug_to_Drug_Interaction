#!/usr/bin/env python3
"""
Drug Interaction CLI Tool
Interactive command-line interface for drug interaction checking
"""

import argparse
import sys
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json
from mt_gat_model import DrugInteractionMTGAT

class DrugInteractionCLI:
    def __init__(self):
        self.model = None
        self.graph_data = None
        self.drug_to_idx = None
        self.idx_to_drug = None
        self.interactions_db = None
        self.drugs_df = None
        self.embeddings = None
        self.load_data()
    
    def load_data(self):
        """Load model and data"""
        print("[+] Loading model and data...")
        
        # Load trained model (prefer v2 format if available)
        model_path = Path('data/trained_model_v2.pt') if Path('data/trained_model_v2.pt').exists() else Path('data/trained_model.pt')
        checkpoint = torch.load(model_path, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {'hidden_dim': 128, 'embedding_dim': 64, 'heads': 4})
            input_dim = checkpoint.get('input_dim', 67)
            self.model = DrugInteractionMTGAT(
                input_dim=input_dim,
                hidden_dim=config.get('hidden_dim', 128),
                embedding_dim=config.get('embedding_dim', 64),
                heads=config.get('heads', 4)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[OK] Model loaded from {model_path}")
        else:
            # Legacy raw state-dict fallback
            self.model = DrugInteractionMTGAT(input_dim=67)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"[OK] Legacy model loaded from {model_path}")

        # Load graph data (prefer v2 graph for v2 model)
        graph_path = Path('data/drug_graph_v2.pt') if Path('data/drug_graph_v2.pt').exists() else Path('data/drug_graph.pt')
        graph_dict = torch.load(graph_path, weights_only=False)
        self.graph_data = graph_dict['graph_data']
        self.drug_to_idx = graph_dict['drug_to_idx']
        self.idx_to_drug = graph_dict['idx_to_drug']
        print(f"[OK] Graph data loaded from {graph_path}")
        
        # Load interactions database
        self.interactions_db = pd.read_csv('data/interactions.csv')
        self.drugs_df = pd.read_csv('data/drugs.csv')
        print("[OK] Database loaded")

        # Cache embeddings once for fast repeated checks.
        with torch.no_grad():
            self.embeddings = self.model.encode(self.graph_data.x, self.graph_data.edge_index)
        print("[OK] Embeddings prepared")
        
        print(f"[INFO] {len(self.drugs_df)} drugs, {len(self.interactions_db)} interactions\n")
    
    def search_drug(self, query):
        """Search for a drug by name or ID"""
        query_lower = query.lower()
        matches = self.drugs_df[
            self.drugs_df['name'].str.lower().str.contains(query_lower, na=False) |
            self.drugs_df['drug_id'].str.lower().str.contains(query_lower, na=False)
        ]
        return matches
    
    def check_interaction(self, drug1_id, drug2_id):
        """Check interaction between two drugs"""
        # Get drug names
        drug1 = self.drugs_df[self.drugs_df['drug_id'] == drug1_id]
        drug2 = self.drugs_df[self.drugs_df['drug_id'] == drug2_id]
        
        if drug1.empty or drug2.empty:
            return None
        
        drug1_name = drug1.iloc[0]['name']
        drug2_name = drug2.iloc[0]['name']
        
        # Check database
        db_interaction = self.interactions_db[
            ((self.interactions_db['drug_1'] == drug1_id) & (self.interactions_db['drug_2'] == drug2_id)) |
            ((self.interactions_db['drug_1'] == drug2_id) & (self.interactions_db['drug_2'] == drug1_id))
        ]
        
        if not db_interaction.empty:
            return {
                'drug1': {'id': drug1_id, 'name': drug1_name},
                'drug2': {'id': drug2_id, 'name': drug2_name},
                'probability': 1.0,
                'risk_level': 'HIGH',
                'source': 'database',
                'description': db_interaction.iloc[0].get('description', 'Known interaction')
            }
        
        # Use MT-GAT
        if drug1_id not in self.drug_to_idx or drug2_id not in self.drug_to_idx:
            return {
                'drug1': {'id': drug1_id, 'name': drug1_name},
                'drug2': {'id': drug2_id, 'name': drug2_name},
                'probability': 0.0,
                'risk_level': 'UNKNOWN',
                'source': 'model',
                'description': 'Drug not in training data'
            }
        
        idx1 = self.drug_to_idx[drug1_id]
        idx2 = self.drug_to_idx[drug2_id]
        
        with torch.no_grad():
            test_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
            # model.decode returns (interaction_logits, severity_logits, confidence)
            int_logit, sev_logit, conf = self.model.decode(self.embeddings, test_edge)
            probability = torch.sigmoid(int_logit).item()
        
        if probability > 0.7:
            risk_level = 'HIGH'
        elif probability > 0.5:
            risk_level = 'MEDIUM'
        elif probability > 0.3:
            risk_level = 'LOW'
        else:
            risk_level = 'VERY LOW'
        
        return {
            'drug1': {'id': drug1_id, 'name': drug1_name},
            'drug2': {'id': drug2_id, 'name': drug2_name},
            'probability': round(probability, 4),
            'risk_level': risk_level,
            'source': 'model',
            'description': f'GAT model prediction: {probability*100:.2f}% probability'
        }
    
    def batch_check(self, drug_ids):
        """Check all pairwise interactions for a list of drugs"""
        results = []
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                result = self.check_interaction(drug_ids[i], drug_ids[j])
                if result and result['probability'] > 0.3:
                    results.append(result)
        return results
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*70)
        print("DRUG INTERACTION CHECKER - INTERACTIVE MODE".center(70))
        print("="*70 + "\n")
        print("Commands:")
        print("  search <query>         - Search for drugs by name")
        print("  check <id1> <id2>      - Check interaction between two drugs")
        print("  batch <id1> <id2> ...  - Check all pairwise interactions")
        print("  exit                   - Exit the program")
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                command = input(">>> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'exit':
                    print("\n[*] Goodbye!")
                    break
                
                elif cmd == 'search':
                    if len(parts) < 2:
                        print("âŒ Usage: search <query>")
                        continue
                    
                    query = ' '.join(parts[1:])
                    results = self.search_drug(query)
                    
                    if results.empty:
                        print(f"[X] No drugs found matching '{query}'")
                    else:
                        print(f"\n[OK] Found {len(results)} drug(s):\n")
                        for idx, row in results.head(10).iterrows():
                            print(f"  {row['drug_id']} - {row['name']}")
                        if len(results) > 10:
                            print(f"\n  ... and {len(results) - 10} more")
                    print()
                
                elif cmd == 'check':
                    if len(parts) < 3:
                        print("âŒ Usage: check <drug_id1> <drug_id2>")
                        continue
                    
                    result = self.check_interaction(parts[1], parts[2])
                    
                    if result is None:
                        print("[X] One or both drugs not found")
                    else:
                        print(f"\n{'='*70}")
                        print(f"Drug Interaction Check")
                        print(f"{'='*70}")
                        print(f"Drug 1: {result['drug1']['name']} ({result['drug1']['id']})")
                        print(f"Drug 2: {result['drug2']['name']} ({result['drug2']['id']})")
                        print(f"Source: {result['source'].upper()}")
                        print(f"Risk Level: {result['risk_level']}")
                        print(f"Probability: {result['probability']*100:.2f}%")
                        print(f"Description: {result['description']}")
                        print(f"{'='*70}\n")
                
                elif cmd == 'batch':
                    if len(parts) < 3:
                        print("âŒ Usage: batch <drug_id1> <drug_id2> [drug_id3] ...")
                        continue
                    
                    drug_ids = parts[1:]
                    results = self.batch_check(drug_ids)
                    
                    if not results:
                        print("[OK] No interactions found")
                    else:
                        print(f"\n[!] Found {len(results)} interaction(s):\n")
                        for result in results:
                            print(f"  {result['drug1']['name']} <-> {result['drug2']['name']}")
                            print(f"    Risk: {result['risk_level']} ({result['probability']*100:.1f}%)")
                            print()
                
                else:
                    print(f"[X] Unknown command: {cmd}")
            
            except KeyboardInterrupt:
                print("\n\n[*] Goodbye!")
                break
            except Exception as e:
                print(f"[X] Error: {str(e)}")
    
    def batch_file_mode(self, input_file, output_file):
        """Process batch file"""
        print(f"\n[+] Processing batch file: {input_file}")
        
        # Read input
        df = pd.read_csv(input_file)
        
        if 'drug_id' not in df.columns and 'drug_ids' not in df.columns:
            print("[X] Error: CSV must have 'drug_id' or 'drug_ids' column")
            return
        
        # Process each row
        all_results = []
        
        if 'drug_ids' in df.columns:
            # Multiple drugs per row (comma-separated)
            for idx, row in df.iterrows():
                drug_ids = str(row['drug_ids']).split(',')
                drug_ids = [d.strip() for d in drug_ids]
                results = self.batch_check(drug_ids)
                all_results.extend(results)
        else:
            # Single drug per row - check against all others
            drug_ids = df['drug_id'].tolist()
            all_results = self.batch_check(drug_ids)
        
        # Export results
        export_data = []
        for result in all_results:
            export_data.append({
                'Drug 1 ID': result['drug1']['id'],
                'Drug 1 Name': result['drug1']['name'],
                'Drug 2 ID': result['drug2']['id'],
                'Drug 2 Name': result['drug2']['name'],
                'Probability': result['probability'],
                'Risk Level': result['risk_level'],
                'Source': result['source'],
                'Description': result['description']
            })
        
        if export_data:
            output_df = pd.DataFrame(export_data)
            output_df.to_csv(output_file, index=False)
            print(f"[OK] Saved {len(export_data)} interactions to {output_file}")
        else:
            print("[OK] No interactions found")

def main():
    parser = argparse.ArgumentParser(description='Drug Interaction Checker CLI')
    parser.add_argument('-i', '--interactive', action='store_true', 
                        help='Run in interactive mode')
    parser.add_argument('-b', '--batch', type=str, 
                        help='Batch mode: input CSV file')
    parser.add_argument('-o', '--output', type=str, default='interactions_output.csv',
                        help='Output file for batch mode')
    parser.add_argument('-c', '--check', nargs=2, metavar=('DRUG1', 'DRUG2'),
                        help='Check interaction between two drugs')
    parser.add_argument('-s', '--search', type=str,
                        help='Search for a drug')
    
    args = parser.parse_args()
    
    try:
        cli = DrugInteractionCLI()
        
        if args.interactive:
            cli.interactive_mode()
        
        elif args.batch:
            cli.batch_file_mode(args.batch, args.output)
        
        elif args.check:
            result = cli.check_interaction(args.check[0], args.check[1])
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Error: Drug not found")
        
        elif args.search:
            results = cli.search_drug(args.search)
            if not results.empty:
                print(results[['drug_id', 'name']].to_string(index=False))
            else:
                print(f"No results found for '{args.search}'")
        
        else:
            # Default to interactive
            cli.interactive_mode()
    
    except Exception as e:
        print(f"[X] Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

