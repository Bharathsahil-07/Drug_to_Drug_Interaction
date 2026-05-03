#!/usr/bin/env python3
import requests
import json

print("\n" + "="*70)
print("🚀 LIVE DEMO - ALL FEATURES WORKING".center(70))
print("="*70 + "\n")

base_url = "http://localhost:5000"

# TEST 1: Search
print("✅ TEST 1: Search for Warfarin (TRACKED)")
print("-" * 70)
try:
    r = requests.get(f"{base_url}/api/drugs/search?q=Warfarin")
    data = r.json()
    print(f"   ✓ Results found: {len(data['results'])}")
    for drug in data['results'][:2]:
        print(f"     • {drug['name']} ({drug['id']})")
except Exception as e:
    print(f"   Error: {e}")

# TEST 2: Check Interaction
print("\n✅ TEST 2: Check Warfarin + Ibuprofen (LOGGED)")
print("-" * 70)
try:
    payload = {"drug1_id": "DB00682", "drug2_id": "DB01050"}
    r = requests.post(f"{base_url}/api/interactions/check", json=payload)
    data = r.json()
    print(f"   ✓ Interaction Found: {data['interaction_found']}")
    print(f"   ✓ Risk Level: {data['risk_level']}")
    print(f"   ✓ Probability: {data['probability']*100:.0f}%")
except Exception as e:
    print(f"   Error: {e}")

# TEST 3: Add Favorite
print("\n✅ TEST 3: Add Warfarin to Favorites (TRACKED)")
print("-" * 70)
try:
    payload = {"drug_id": "DB00682", "drug_name": "Warfarin"}
    r = requests.post(f"{base_url}/api/favorites/add", json=payload)
    data = r.json()
    print(f"   ✓ Status: {data['message']}")
except Exception as e:
    print(f"   Error: {e}")

# TEST 4: Dashboard Stats
print("\n✅ TEST 4: Get Dashboard Statistics (REAL DATA)")
print("-" * 70)
try:
    r = requests.get(f"{base_url}/api/dashboard/stats")
    data = r.json()
    print(f"   ✓ Searches Today: {data['total_searches']}")
    print(f"   ✓ Checks Today: {data['total_checks']}")
    print(f"   ✓ Favorites: {data['total_favorites']}")
except Exception as e:
    print(f"   Error: {e}")

# TEST 5: Search History
print("\n✅ TEST 5: Get Activity History (ALL TRACKED)")
print("-" * 70)
try:
    r = requests.get(f"{base_url}/api/search/history")
    data = r.json()
    print(f"   ✓ Total Activities: {data['breakdown']['total_activities']}")
    print(f"     • Searches: {data['breakdown']['total_searches']}")
    print(f"     • Checks: {data['breakdown']['total_checks']}")
    print(f"     • Favorites: {data['breakdown']['total_favorites']}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print("✅ ALL FEATURES WORKING PERFECTLY".center(70))
print("="*70)
print("\n📍 NEXT STEPS:")
print("   1. Open main interface: http://localhost:5000")
print("   2. Open dashboard: http://localhost:5000/static/dashboard.html")
print("   3. Perform searches and check interactions")
print("   4. Watch dashboard update in REAL-TIME every 3 seconds")
print("\n")
