import xml.etree.ElementTree as ET
import os

# Parse the large XML file
xml_file = "full database.xml"
print(f"Parsing {xml_file}...")

test_drugs = ["DB12445", "DB00003"]
found_data = {}

try:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find drug namespace
    ns = {'db': 'http://www.drugbank.ca'}
    if not root.tag.startswith('{'):
        ns = {}
    
    # Iterate through drugs
    drugs = root.findall('.//db:drug', ns) if ns else root.findall('.//drug')
    total_drugs = len(drugs)
    print(f"Total drugs in XML: {total_drugs}")
    
    for i, drug in enumerate(drugs):
        if i % 2000 == 0:
            print(f"Processing drug {i}/{total_drugs}...", flush=True)
        
        # Get drug ID
        drug_id_elem = drug.find('db:drugbank-id', ns) if ns else drug.find('drugbank-id')
        if drug_id_elem is None:
            drug_id_elem = drug.find('.//db:drugbank-id[@primary="true"]', ns) if ns else drug.find('.//drugbank-id[@primary="true"]')
        
        if drug_id_elem is not None:
            drug_id = drug_id_elem.text
            
            if drug_id in test_drugs:
                print(f"\n✓ Found {drug_id}")
                
                # Get name
                name_elem = drug.find('db:name', ns) if ns else drug.find('name')
                name = name_elem.text if name_elem is not None else "N/A"
                
                # Get SMILES
                smiles_elem = drug.find('.//db:smiles', ns) if ns else drug.find('.//smiles')
                smiles = smiles_elem.text if smiles_elem is not None else None
                
                # Get inchi
                inchi_elem = drug.find('.//db:inchi', ns) if ns else drug.find('.//inchi')
                inchi = inchi_elem.text if inchi_elem is not None else None
                
                # Get ATC codes
                atc_elems = drug.findall('.//db:atc-code', ns) if ns else drug.findall('.//atc-code')
                atc_codes = [elem.text for elem in atc_elems if elem.text]
                
                found_data[drug_id] = {
                    'name': name,
                    'smiles': smiles,
                    'inchi': inchi,
                    'atc_codes': atc_codes
                }
    
    print(f"\n\n=== RESULTS ===")
    for drug_id in test_drugs:
        if drug_id in found_data:
            data = found_data[drug_id]
            print(f"\n{drug_id} ({data['name']}):")
            print(f"  SMILES: {data['smiles'] if data['smiles'] else '❌ NOT FOUND'}")
            print(f"  InChI: {data['inchi'][:80] + '...' if data['inchi'] and len(data['inchi']) > 80 else data['inchi'] if data['inchi'] else '❌ NOT FOUND'}")
            print(f"  ATC Codes: {data['atc_codes'] if data['atc_codes'] else '❌ NOT FOUND'}")
        else:
            print(f"\n{drug_id}: ❌ NOT FOUND in XML")

except Exception as e:
    print(f"Error parsing XML: {e}")
    import traceback
    traceback.print_exc()
