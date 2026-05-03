import xml.etree.ElementTree as ET

# Check a variety of drugs to see if ANY have SMILES/ATC
xml_file = "full database.xml"
sample_count = 0
with_smiles = 0
with_atc = 0
drugs_checked = 0

print("Sampling random drugs from XML to check data availability...")

try:
    for event, elem in ET.iterparse(xml_file, events=['end']):
        if elem.tag.endswith('drug'):
            drugs_checked += 1
            
            # Every 500th drug, sample it
            if drugs_checked % 500 == 0:
                # Get drugbank-id
                drug_id_elem = elem.find('.//{http://www.drugbank.ca}drugbank-id[@primary="true"]')
                if drug_id_elem is None:
                    drug_id_elem = elem.find('.//{http://www.drugbank.ca}drugbank-id')
                
                if drug_id_elem is not None:
                    drug_id = drug_id_elem.text
                    name_elem = elem.find('{http://www.drugbank.ca}name')
                    name = name_elem.text if name_elem is not None else "N/A"
                    
                    smiles_elem = elem.find('.//{http://www.drugbank.ca}smiles')
                    has_smiles = smiles_elem is not None and smiles_elem.text
                    
                    atc_elems = elem.findall('.//{http://www.drugbank.ca}atc-code')
                    has_atc = any(e.text for e in atc_elems)
                    
                    sample_count += 1
                    if has_smiles:
                        with_smiles += 1
                    if has_atc:
                        with_atc += 1
                    
                    print("  Drug {}: {} - SMILES: {} | ATC: {}".format(
                        drug_id, name[:40], "YES" if has_smiles else "NO", "YES" if has_atc else "NO"))
                    
                    if sample_count >= 20:
                        break
            
            # Clear element to free memory
            elem.clear()
    
    print("\n=== SAMPLE STATISTICS ===")
    print("Drugs sampled: {}".format(sample_count))
    print("With SMILES: {} ({:.1f}%)".format(with_smiles, 100*with_smiles/sample_count if sample_count > 0 else 0))
    print("With ATC codes: {} ({:.1f}%)".format(with_atc, 100*with_atc/sample_count if sample_count > 0 else 0))
    print("Total drugs processed: {}".format(drugs_checked))

except Exception as e:
    print("Error: {}".format(e))
    import traceback
    traceback.print_exc()
