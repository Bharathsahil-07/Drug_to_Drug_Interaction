import xml.etree.ElementTree as ET

# Fast streaming parser
xml_file = "full database.xml"
test_drugs = {"DB12445", "DB00003"}
found_data = {}

print("Fast scanning {}...".format(xml_file))

try:
    for event, elem in ET.iterparse(xml_file, events=['end']):
        if elem.tag.endswith('drug'):
            # Get drugbank-id
            drug_id_elem = elem.find('.//{http://www.drugbank.ca}drugbank-id[@primary="true"]')
            if drug_id_elem is None:
                drug_id_elem = elem.find('.//{http://www.drugbank.ca}drugbank-id')
            
            if drug_id_elem is not None and drug_id_elem.text in test_drugs:
                drug_id = drug_id_elem.text
                print("\nFOUND: {}".format(drug_id))
                
                # Get name
                name_elem = elem.find('{http://www.drugbank.ca}name')
                name = name_elem.text if name_elem is not None else "N/A"
                
                # Get SMILES
                smiles_elem = elem.find('.//{http://www.drugbank.ca}smiles')
                smiles = smiles_elem.text if smiles_elem is not None else None
                
                # Get InChI
                inchi_elem = elem.find('.//{http://www.drugbank.ca}inchi')
                inchi = inchi_elem.text if inchi_elem is not None else None
                
                # Get ATC codes
                atc_elems = elem.findall('.//{http://www.drugbank.ca}atc-code')
                atc_codes = [e.text for e in atc_elems if e.text]
                
                found_data[drug_id] = {
                    'name': name,
                    'smiles': smiles,
                    'inchi': inchi,
                    'atc_codes': atc_codes
                }
                
                test_drugs.discard(drug_id)
                if not test_drugs:
                    break
            
            # Clear element to free memory
            elem.clear()
    
    print("\n\n=== RESULTS ===")
    if "DB00003" in found_data:
        data = found_data["DB00003"]
        print("\nDB00003 (Dornase Alfa):")
        print("  Name: {}".format(data['name']))
        print("  SMILES: {}".format(data['smiles'] if data['smiles'] else "NOT FOUND"))
        inchi_disp = data['inchi'][:80] + '...' if data['inchi'] and len(data['inchi']) > 80 else (data['inchi'] if data['inchi'] else "NOT FOUND")
        print("  InChI: {}".format(inchi_disp))
        print("  ATC Codes: {}".format(', '.join(data['atc_codes']) if data['atc_codes'] else "NOT FOUND"))
    else:
        print("\nDB00003: NOT FOUND in XML")
    
    if "DB12445" in found_data:
        data = found_data["DB12445"]
        print("\nDB12445 (Nitroaspirin):")
        print("  Name: {}".format(data['name']))
        print("  SMILES: {}".format(data['smiles'] if data['smiles'] else "NOT FOUND"))
        inchi_disp = data['inchi'][:80] + '...' if data['inchi'] and len(data['inchi']) > 80 else (data['inchi'] if data['inchi'] else "NOT FOUND")
        print("  InChI: {}".format(inchi_disp))
        print("  ATC Codes: {}".format(', '.join(data['atc_codes']) if data['atc_codes'] else "NOT FOUND"))
    else:
        print("\nDB12445: NOT FOUND in XML")
    
    if not found_data:
        print("\nNeither drug found in XML")

except Exception as e:
    print("Error: {}".format(e))
    import traceback
    traceback.print_exc()
