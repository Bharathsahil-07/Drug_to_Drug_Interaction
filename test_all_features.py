"""
Test All New Features
Execute this script to verify all enhancements are working
"""

import requests
import json
import time

API_URL = "http://localhost:5000"

def test_feature(name, func):
    """Test wrapper"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    try:
        func()
        print(f"[PASS] {name} - PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {name} - FAILED: {str(e)}")
        return False

def test_health_check():
    """Test basic health check"""
    response = requests.get(f"{API_URL}/api/health")
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Model loaded: {data['model_loaded']}")
    print(f"Drugs: {data['num_drugs']}")
    print(f"Interactions: {data['num_interactions']}")
    assert data['model_loaded'] == True

def test_enhanced_stats():
    """Test enhanced statistics"""
    response = requests.get(f"{API_URL}/api/stats")
    data = response.json()
    print(f"Total drugs: {data['total_drugs']}")
    print(f"Total interactions: {data['total_interactions']}")
    print(f"High risk: {data.get('high_risk_count', 'N/A')}")
    print(f"Search history: {data.get('search_history_count', 0)}")
    assert 'high_risk_count' in data

def test_severity_analysis():
    """Test severity analysis endpoint"""
    payload = {
        "drug1": "DB00682",  # Warfarin
        "drug2": "DB01050"   # Ibuprofen
    }
    response = requests.post(f"{API_URL}/api/interactions/severity", json=payload)
    data = response.json()
    print(f"Severity: {data.get('severity', 'N/A')}")
    print(f"Clinical action: {data.get('clinical_action', 'N/A')}")
    print(f"Probability: {data.get('probability', 0)*100:.1f}%")
    assert 'severity' in data

def test_alternative_suggestions():
    """Test alternative drug suggestions"""
    payload = {
        "drug_id": "DB00682",  # Warfarin
        "interacting_with": ["DB01050"]  # Ibuprofen
    }
    response = requests.post(f"{API_URL}/api/alternatives/suggest", json=payload)
    data = response.json()
    print(f"Original drug: {data['original_drug']['name']}")
    print(f"Alternatives found: {data['total_found']}")
    if data['alternatives']:
        print(f"Top alternative: {data['alternatives'][0]['name']}")

def test_patient_profile():
    """Test patient profile creation"""
    payload = {
        "patient_id": "P12345",
        "medications": ["DB00682", "DB01050", "DB00945"]  # Warfarin, Ibuprofen, Aspirin
    }
    response = requests.post(f"{API_URL}/api/patient/profile", json=payload)
    data = response.json()
    print(f"Patient ID: {data['patient_id']}")
    print(f"Total medications: {data['total_medications']}")
    print(f"Interactions found: {data['interactions_found']}")
    assert data['profile_created'] == True

def test_get_patient_profile():
    """Test retrieving patient profile"""
    response = requests.get(f"{API_URL}/api/patient/profile/P12345")
    data = response.json()
    print(f"Patient medications: {data['current_analysis']['total_medications']}")
    print(f"High risk interactions: {data['current_analysis']['high_risk']}")
    print(f"Medium risk interactions: {data['current_analysis']['medium_risk']}")

def test_batch_export():
    """Test batch interaction export"""
    payload = {
        "interactions": [
            {
                "drug1": {"id": "DB00682", "name": "Warfarin"},
                "drug2": {"id": "DB01050", "name": "Ibuprofen"},
                "probability": 0.95,
                "risk_level": "HIGH",
                "source": "database",
                "description": "Test interaction"
            }
        ]
    }
    response = requests.post(f"{API_URL}/api/export/csv", json=payload)
    print(f"Export status: {response.status_code}")
    print(f"Content type: {response.headers.get('Content-Type')}")
    assert response.status_code == 200

def test_report_generation():
    """Test report generation"""
    payload = {
        "patient_id": "P12345",
        "interactions": [
            {
                "drug1": {"id": "DB00682", "name": "Warfarin"},
                "drug2": {"id": "DB01050", "name": "Ibuprofen"},
                "probability": 0.95,
                "risk_level": "HIGH"
            }
        ]
    }
    response = requests.post(f"{API_URL}/api/export/report", json=payload)
    data = response.json()
    print(f"Report ID: {data['report_id']}")
    print(f"Total interactions: {data['summary']['total_interactions']}")
    print(f"Severe: {data['summary']['severe']}")
    assert 'report_id' in data

def test_dashboard_stats():
    """Test dashboard statistics"""
    response = requests.get(f"{API_URL}/api/dashboard/stats")
    data = response.json()
    print(f"Overview drugs: {data['overview']['total_drugs']}")
    print(f"Model accuracy: {data['overview']['model_accuracy']*100:.1f}%")
    print(f"Dangerous combos found: {len(data['top_dangerous_combinations'])}")
    assert 'overview' in data
    assert 'model_performance' in data

def test_model_metrics():
    """Test model metrics endpoint"""
    response = requests.get(f"{API_URL}/api/model/metrics")
    data = response.json()
    print(f"Architecture: {data['architecture']['type']}")
    print(f"Layers: {data['architecture']['layers']}")
    print(f"AUC Score: {data['performance']['auc_score']*100:.2f}%")
    print(f"Total parameters: {data['architecture']['total_parameters']:,}")
    assert data['performance']['auc_score'] > 0.9

def test_search_history():
    """Test search history endpoint"""
    response = requests.get(f"{API_URL}/api/search/history")
    data = response.json()
    print(f"History entries: {data['total']}")
    assert 'history' in data

def main():
    print("\n" + "="*70)
    print("TESTING ALL NEW FEATURES".center(70))
    print("="*70)
    print("\nMake sure the API server is running at http://localhost:5000")
    print("\nWaiting 2 seconds...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Enhanced Statistics", test_enhanced_stats),
        ("Severity Analysis", test_severity_analysis),
        ("Alternative Suggestions", test_alternative_suggestions),
        ("Patient Profile Creation", test_patient_profile),
        ("Get Patient Profile", test_get_patient_profile),
        ("Batch Export CSV", test_batch_export),
        ("Report Generation", test_report_generation),
        ("Dashboard Statistics", test_dashboard_stats),
        ("Model Metrics", test_model_metrics),
        ("Search History", test_search_history),
    ]
    
    passed = 0
    failed = 0
    
    for name, func in tests:
        if test_feature(name, func):
            passed += 1
        else:
            failed += 1
        time.sleep(0.5)
    
    print("\n" + "="*70)
    print("TEST RESULTS".center(70))
    print("="*70)
    print(f"[PASS] Passed: {passed}/{len(tests)}")
    print(f"[FAIL] Failed: {failed}/{len(tests)}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
