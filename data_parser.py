"""
Drug Interaction Project - XML Parser
Extracts drug and interaction data from DrugBank XML file
"""

import xml.etree.ElementTree as ET
import pandas as pd
import json
from tqdm import tqdm
import re

class DrugBankParser:
    """Parse DrugBank XML and extract relevant data"""
    
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.namespace = {'db': 'http://www.drugbank.ca'}
        
    def parse_all(self, max_drugs=None):
        """
        Parse entire XML file and extract drugs and interactions
        
        Args:
            max_drugs: Limit number of drugs to parse (for testing)
        
        Returns:
            drugs_df: DataFrame with drug information
            interactions_df: DataFrame with drug interactions
        """
        print(f"Parsing {self.xml_file}...")
        print("This may take a few minutes for large files...")
        
        # Parse XML iteratively (memory efficient for large files)
        drugs_data = []
        interactions_data = []
        
        context = ET.iterparse(self.xml_file, events=('end',))
        
        drug_count = 0
        skipped_count = 0
        for event, elem in context:
            if elem.tag == '{http://www.drugbank.ca}drug':
                drug_count += 1
                
                # Extract drug info
                drug_info = self._extract_drug_info(elem)
                
                # Skip if extraction failed (no valid ID or name)
                if drug_info is None:
                    skipped_count += 1
                    elem.clear()
                    continue
                
                drugs_data.append(drug_info)
                
                # Extract interactions for this drug
                drug_interactions = self._extract_interactions(elem, drug_info['drug_id'])
                interactions_data.extend(drug_interactions)
                
                # Clear element to free memory
                elem.clear()
                
                if drug_count % 100 == 0:
                    print(f"Processed {drug_count} drugs...")
                
                if max_drugs and drug_count >= max_drugs:
                    break
        
        print(f"\n✅ Completed!")
        print(f"   Total processed: {drug_count} drug entries")
        print(f"   Successfully extracted: {len(drugs_data)} drugs")
        print(f"   Skipped (incomplete): {skipped_count} entries")
        print(f"   Total interactions: {len(interactions_data)}")
        
        # Convert to DataFrames
        drugs_df = pd.DataFrame(drugs_data)
        interactions_df = pd.DataFrame(interactions_data)
        
        # Remove duplicates in interactions (A-B and B-A are same)
        interactions_df = self._remove_duplicate_interactions(interactions_df)
        
        return drugs_df, interactions_df
    
    def _extract_drug_info(self, drug_elem):
        """Extract relevant information for a single drug"""
        ns = self.namespace
        
        # Primary DrugBank ID - try multiple methods to get it
        drug_id = None
        
        # Method 1: Find primary drugbank-id
        drug_id_elem = drug_elem.find(".//db:drugbank-id[@primary='true']", ns)
        if drug_id_elem is not None:
            drug_id = drug_id_elem.text
        
        # Method 2: If no primary, get first drugbank-id
        if not drug_id:
            drug_id_elem = drug_elem.find(".//db:drugbank-id", ns)
            if drug_id_elem is not None:
                drug_id = drug_id_elem.text
        
        # Method 3: Check direct child drugbank-id elements
        if not drug_id:
            for db_id in drug_elem.findall('db:drugbank-id', ns):
                if db_id.text and db_id.text.startswith('DB'):
                    drug_id = db_id.text
                    break
        
        # Skip this drug if no valid ID found
        if not drug_id:
            return None
        
        # Basic info
        name = self._get_text(drug_elem, 'db:name', ns)
        
        # Skip if no name either
        if not name:
            return None
            
        description = self._get_text(drug_elem, 'db:description', ns)
        cas_number = self._get_text(drug_elem, 'db:cas-number', ns)
        
        # Drug type and state
        drug_type = drug_elem.get('type', '')
        state = self._get_text(drug_elem, 'db:state', ns)
        
        # Clinical info
        indication = self._get_text(drug_elem, 'db:indication', ns)
        pharmacodynamics = self._get_text(drug_elem, 'db:pharmacodynamics', ns)
        mechanism = self._get_text(drug_elem, 'db:mechanism-of-action', ns)
        toxicity = self._get_text(drug_elem, 'db:toxicity', ns)
        
        # Pharmacokinetics
        absorption = self._get_text(drug_elem, 'db:absorption', ns)
        metabolism = self._get_text(drug_elem, 'db:metabolism', ns)
        half_life = self._get_text(drug_elem, 'db:half-life', ns)
        protein_binding = self._get_text(drug_elem, 'db:protein-binding', ns)
        
        # Categories
        categories = []
        for cat in drug_elem.findall('.//db:category/db:category', ns):
            if cat.text:
                categories.append(cat.text)
        
        # Groups (approved, experimental, etc.)
        groups = []
        for group in drug_elem.findall('.//db:group', ns):
            if group.text:
                groups.append(group.text)
        
        # Quality check: Skip entries with insufficient data
        # Must have at least one of: description, drug_type, indication, or groups
        has_meaningful_data = any([
            description,
            drug_type,
            indication,
            len(groups) > 0,
            len(categories) > 0
        ])
        
        if not has_meaningful_data:
            # This is likely a stub/reference entry - skip it
            return None
        
        return {
            'drug_id': drug_id,
            'name': name,
            'description': description,
            'cas_number': cas_number,
            'drug_type': drug_type,
            'state': state,
            'indication': indication,
            'pharmacodynamics': pharmacodynamics,
            'mechanism_of_action': mechanism,
            'toxicity': toxicity,
            'absorption': absorption,
            'metabolism': metabolism,
            'half_life': half_life,
            'protein_binding': protein_binding,
            'categories': '|'.join(categories),
            'groups': '|'.join(groups)
        }
    
    def _extract_interactions(self, drug_elem, drug_id):
        """Extract all interactions for a drug"""
        ns = self.namespace
        interactions = []
        
        # Skip if drug_id is None
        if not drug_id:
            return interactions
        
        for interaction in drug_elem.findall('.//db:drug-interaction', ns):
            interacting_drug_id = self._get_text(interaction, 'db:drugbank-id', ns)
            interacting_drug_name = self._get_text(interaction, 'db:name', ns)
            description = self._get_text(interaction, 'db:description', ns)
            
            # Only add if we have both drug ID and description
            if interacting_drug_id and description and interacting_drug_id.startswith('DB'):
                interactions.append({
                    'drug_1': drug_id,
                    'drug_2': interacting_drug_id,
                    'drug_2_name': interacting_drug_name,
                    'description': description
                })
        
        return interactions
    
    def _get_text(self, parent, tag, namespace):
        """Safely get text from XML element"""
        elem = parent.find(tag, namespace)
        if elem is not None and elem.text:
            # Clean text
            text = elem.text.strip()
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            return text
        return None
    
    def _remove_duplicate_interactions(self, interactions_df):
        """Remove duplicate interactions (A-B same as B-A)"""
        if len(interactions_df) == 0:
            return interactions_df
            
        # Sort drug IDs so A-B and B-A become same
        interactions_df['drug_pair'] = interactions_df.apply(
            lambda x: tuple(sorted([x['drug_1'], x['drug_2']])), 
            axis=1
        )
        
        # Remove duplicates
        interactions_df = interactions_df.drop_duplicates(subset=['drug_pair'])
        interactions_df = interactions_df.drop(columns=['drug_pair'])
        
        return interactions_df
    
    def save_to_csv(self, drugs_df, interactions_df, output_dir='data'):
        """Save parsed data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        drugs_file = os.path.join(output_dir, 'drugs.csv')
        interactions_file = os.path.join(output_dir, 'interactions.csv')
        
        drugs_df.to_csv(drugs_file, index=False)
        interactions_df.to_csv(interactions_file, index=False)
        
        print(f"\nSaved data to:")
        print(f"  - {drugs_file} ({len(drugs_df)} drugs)")
        print(f"  - {interactions_file} ({len(interactions_df)} interactions)")
        
        return drugs_file, interactions_file


def main():
    """Example usage"""
    import os
    
    # Path to your XML file
    xml_file = r"c:\Users\navas\Downloads\Sdp\full database.xml"
    
    # Check if file exists
    if not os.path.exists(xml_file):
        print(f"Error: File not found: {xml_file}")
        return
    
    # Parse the XML (limit to 1000 drugs for testing)
    parser = DrugBankParser(xml_file)
    
    print("Starting parse... (use max_drugs=100 for quick test)")
    choice = input("Parse all drugs? (y/n, default=100 for testing): ").lower()
    
    if choice == 'y':
        drugs_df, interactions_df = parser.parse_all()
    else:
        drugs_df, interactions_df = parser.parse_all(max_drugs=100)
    
    # Display summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total drugs: {len(drugs_df)}")
    print(f"Total interactions: {len(interactions_df)}")
    print(f"\nDrug types:")
    print(drugs_df['drug_type'].value_counts())
    print(f"\nSample drugs:")
    print(drugs_df[['drug_id', 'name', 'drug_type']].head(10))
    
    # Save to CSV
    parser.save_to_csv(drugs_df, interactions_df)
    
    print("\n✅ Parsing complete! Data ready for graph construction.")


if __name__ == "__main__":
    main()
