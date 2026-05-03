import requests
import json

try:
    print("Testing /api/graph/3d endpoint...")
    response = requests.get('http://localhost:5000/api/graph/3d?limit=10')
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Success!")
        print(f"Nodes: {len(data['nodes'])}")
        print(f"Links: {len(data['links'])}")
        print("Sample Node:", data['nodes'][0] if data['nodes'] else "None")
        print("Sample Link:", data['links'][0] if data['links'] else "None")
    else:
        print(f"❌ Failed with status {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")
