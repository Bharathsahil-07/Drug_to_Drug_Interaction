import pandas as pd

drugs = pd.read_csv('data/drugs.csv', dtype=str)  # Read as strings to preserve NaN

# Search case-insensitive
drugs['name_lower'] = drugs['name'].str.lower()

print("Searching for 'aspirin'...")
aspirin = drugs[drugs['name_lower'].str.contains('aspirin', na=False)]
print(f"Found {len(aspirin)} matches:")
for drug_id, name in zip(aspirin['drug_id'], aspirin['name']):
    print(f"  {drug_id}: {name}")

print("\n" + "="*60 + "\n")

print("Searching for 'warfarin'...")
warfarin = drugs[drugs['name_lower'].str.contains('warfarin', na=False)]
print(f"Found {len(warfarin)} matches:")
for drug_id, name in zip(warfarin['drug_id'], warfarin['name']):
    print(f"  {drug_id}: {name}")

print("\n" + "="*60 + "\n")

# Try searching interactions
print("Searching interactions for aspirin-warfarin combinations...")
interactions = pd.read_csv('data/interactions.csv')

aspirin_ids = aspirin[aspirin['drug_id'] != 'nan']['drug_id'].tolist()
warfarin_ids = warfarin[warfarin['drug_id'] != 'nan']['drug_id'].tolist()

if aspirin_ids and warfarin_ids:
    print(f"Aspirin IDs: {aspirin_ids}")
    print(f"Warfarin IDs: {warfarin_ids}")
    
    count = 0
    for asp_id in aspirin_ids:
        for war_id in warfarin_ids:
            inter = interactions[
                ((interactions['drug_id_1'] == asp_id) & (interactions['drug_id_2'] == war_id)) |
                ((interactions['drug_id_1'] == war_id) & (interactions['drug_id_2'] == asp_id))
            ]
            if not inter.empty:
                count += 1
                desc = inter.iloc[0]['description']
                print(f"\n✅ {asp_id} + {war_id}:")
                print(f"   {desc[:150]}")
    
    if count == 0:
        print("No interactions found between these drug combinations")
else:
    print("Could not find drug IDs for search")
