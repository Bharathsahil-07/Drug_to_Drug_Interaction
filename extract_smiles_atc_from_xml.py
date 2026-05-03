import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

NAMESPACE = {"db": "http://www.drugbank.ca"}


def get_primary_drugbank_id(drug_elem):
    primary = drug_elem.find("db:drugbank-id[@primary='true']", NAMESPACE)
    if primary is not None and primary.text:
        return primary.text.strip()

    fallback = drug_elem.find("db:drugbank-id", NAMESPACE)
    if fallback is not None and fallback.text:
        return fallback.text.strip()

    return None


def get_drug_name(drug_elem):
    name_elem = drug_elem.find("db:name", NAMESPACE)
    return name_elem.text.strip() if name_elem is not None and name_elem.text else None


def get_property_value(drug_elem, target_kind):
    # DrugBank stores these under calculated-properties and sometimes experimental-properties.
    for base_path in (
        "db:calculated-properties/db:property",
        "db:experimental-properties/db:property",
    ):
        for prop in drug_elem.findall(base_path, NAMESPACE):
            kind_elem = prop.find("db:kind", NAMESPACE)
            value_elem = prop.find("db:value", NAMESPACE)
            if kind_elem is None or value_elem is None:
                continue

            if not kind_elem.text or not value_elem.text:
                continue

            if kind_elem.text.strip().upper() == target_kind.upper():
                return value_elem.text.strip()

    return None


def get_atc_codes(drug_elem):
    codes = []
    for atc_elem in drug_elem.findall("db:atc-codes/db:atc-code", NAMESPACE):
        code_attr = atc_elem.attrib.get("code")
        if code_attr:
            code = code_attr.strip()
            if code:
                codes.append(code)

    return sorted(set(codes))


def extract_records(xml_path, include_missing=False, limit=None):
    records = []
    processed = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if not elem.tag.endswith("drug"):
            continue

        drug_id = get_primary_drugbank_id(elem)
        name = get_drug_name(elem)
        smiles = get_property_value(elem, "SMILES")
        inchi = get_property_value(elem, "InChI")
        inchikey = get_property_value(elem, "InChIKey")
        atc_codes = get_atc_codes(elem)

        if include_missing or smiles:
            records.append(
                {
                    "drug_id": drug_id,
                    "name": name,
                    "smiles": smiles,
                    "inchi": inchi,
                    "inchikey": inchikey,
                    "atc_codes": ";".join(atc_codes) if atc_codes else None,
                }
            )

        processed += 1
        if limit is not None and processed >= limit:
            elem.clear()
            break

        elem.clear()

    return records


def write_csv(records, output_csv):
    fieldnames = ["drug_id", "name", "smiles", "inchi", "inchikey", "atc_codes"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DrugBank drug_id, name, SMILES/InChI/InChIKey, and ATC codes from XML"
    )
    parser.add_argument(
        "--xml",
        default="full database.xml",
        help="Path to DrugBank XML file (default: full database.xml)",
    )
    parser.add_argument(
        "--out",
        default="drugbank_smiles_atc.csv",
        help="Output CSV path (default: drugbank_smiles_atc.csv)",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include drugs even when SMILES is missing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of <drug> entries to process (for testing)",
    )

    args = parser.parse_args()
    xml_path = Path(args.xml)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    records = extract_records(
        xml_path=xml_path,
        include_missing=args.include_missing,
        limit=args.limit,
    )
    write_csv(records, args.out)

    with_smiles = sum(1 for r in records if r.get("smiles"))
    with_atc = sum(1 for r in records if r.get("atc_codes"))

    print(f"Done. Wrote {len(records)} rows to {args.out}")
    print(f"Rows with SMILES: {with_smiles}")
    print(f"Rows with ATC codes: {with_atc}")


if __name__ == "__main__":
    main()
