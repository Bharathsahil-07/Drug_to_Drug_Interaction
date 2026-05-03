import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_api():
    print(f"Testing API at {BASE_URL}...")
    
    # 1. Health Check
    try:
        response = requests.get(f'{BASE_URL}/api/health')
        print(f"Health Check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return

    # 2. Insights
    try:
        response = requests.get(f'{BASE_URL}/api/insights')
        print(f"Insights: {response.status_code}")
        # print(response.json())
    except Exception as e:
        print(f"Insights Failed: {e}")

    # 3. Batch Interactions
    payload = {
        "drugs": ["DB00001", "DB00006"] # Lepirudin vs Bivalirudin (both antithrombotics, might interact or at least run)
    }
    
    print("\nTesting Batch Interactions...")
    try:
        response = requests.post(
            f'{BASE_URL}/api/interactions/batch',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Batch Interaction Failed: {e}")

if __name__ == "__main__":
    test_api()
