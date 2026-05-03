import requests
import json

try:
    print("Testing /api/graph/3d endpoint with limit=2000...")
    response = requests.get('http://localhost:5000/api/graph/3d?limit=2000')
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Success!")
        print(f"Nodes: {len(data['nodes'])}")
        print(f"Links: {len(data['links'])}")
        
        if len(data['links']) <= 5000:
             print("✅ Edge limit respected!")
        else:
             print(f"❌ Edge limit exceeded: {len(data['links'])}")

    else:
        print(f"❌ Failed with status {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")
