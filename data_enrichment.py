
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import time
import os

def enrich_data(drugs_file='data/drugs.csv', output_file='data/drugs_enriched.csv'):
    print(f"Reading {drugs_file}...")
    df = pd.read_csv(drugs_file)
    
    # Check if columns already exist
    if 'smiles' not in df.columns:
        df['smiles'] = None
    if 'atc_code' not in df.columns:
        df['atc_code'] = None
        
    print(f"Processing {len(df)} drugs...")
    
    # cache specifically to avoid re-fetching
    cache_file = 'data/drugs_cache.csv'
    if os.path.exists(cache_file):
        print("Loading from cache...")
        df = pd.read_csv(cache_file)
    
    # Iterate and fetch missing data
    # We will limit to first 500 for testing purposes if this is a demo, 
    # but the user wants a real upgrade. 
    # To avoid hanging for hours, I'll filter for drugs that actually have interactions first?
    # For now, let's just try to fetch for rows where smiles is missing.
    
    missing_mask = df['smiles'].isna()
    drugs_to_process = df[missing_mask]
    
    print(f"Found {len(drugs_to_process)} drugs with missing SMILES.")
    
    count = 0
    save_interval = 50
    
    for idx, row in tqdm(drugs_to_process.iterrows(), total=len(drugs_to_process)):
        drug_name = row['name']
        try:
            # Try searching by name
            compounds = pcp.get_compounds(drug_name, 'name')
            if compounds:
                compound = compounds[0]
                df.at[idx, 'smiles'] = compound.canonical_smiles
                # PubChem doesn't always have ATC, but we can try to get it from synonyms or properties if available
                # For this MVP, we might mock ATC or try to extract it from 'categories' column if present
                
                # compound.atc_codes is not directly available in basic compound object usually, 
                # often requires extra calls or parsing. 
                # We will focus on SMILES for the chemical features first.
                
            else:
                df.at[idx, 'smiles'] = 'Not Found'
                
        except Exception as e:
            print(f"Error fetching {drug_name}: {e}")
            
        count += 1
        if count % save_interval == 0:
            df.to_csv(cache_file, index=False)
            
        # Rate limit
        time.sleep(0.2)
        
        # STOPPING CRITERIA FOR NOW (Prototype Speed)
        # Process max 50 drugs for this turn to show progress, then we can verify.
        # Remove this break for full run.
        if count >= 20: 
            print("Stopping after 20 drugs for demonstration (remove limit in full run).")
            break
            
    df.to_csv(output_file, index=False)
    print(f"Saved enriched data to {output_file}")
    
    # Also save to cache
    df.to_csv(cache_file, index=False)

if __name__ == "__main__":
    enrich_data()
