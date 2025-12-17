"""
Test Client for Air Quality Prediction API
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("=" * 80)
    print("TESTING HEALTH CHECK ENDPOINT")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    print("=" * 80)
    print("TESTING MODEL INFO ENDPOINT")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()

def test_prediction(endpoint: str, model_name: str):
    """Test a prediction endpoint"""
    print("=" * 80)
    print(f"TESTING {model_name} PREDICTION")
    print("=" * 80)
    
    # Sample input data
    input_data = {
        "PT08_S1CO": 1200.0,
        "NMHCGT": 150.0,
        "C6H6GT": 10.5,
        "PT08_S2NMHC": 950.0,
        "NOxGT": 200.0,
        "PT08_S3NOx": 800.0,
        "NO2GT": 120.0,
        "PT08_S4NO2": 1500.0,
        "PT08_S5O3": 1100.0,
        "T": 20.0,
        "RH": 50.0,
        "AH": 1.0,
        "hour": 12,
        "day_of_week": 3,
        "month": 6,
        "is_weekend": 0
    }
    
    print(f"\nInput Data:\n{json.dumps(input_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/{endpoint}",
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  Predicted CO(GT): {result['prediction']:.4f}")
        print(f"  Model Name: {result['model_name']}")
        print(f"  Model Version: {result['model_version']}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"Error: {response.text}")
    
    print()

def test_batch_predictions():
    """Test predictions with multiple scenarios"""
    print("=" * 80)
    print("TESTING BATCH PREDICTIONS (Multiple Scenarios)")
    print("=" * 80)
    
    scenarios = [
        {
            "name": "Low Pollution",
            "data": {
                "PT08_S1CO": 800.0,
                "NMHCGT": 100.0,
                "C6H6GT": 5.0,
                "PT08_S2NMHC": 700.0,
                "NOxGT": 100.0,
                "PT08_S3NOx": 600.0,
                "NO2GT": 80.0,
                "PT08_S4NO2": 1200.0,
                "PT08_S5O3": 900.0,
                "T": 15.0,
                "RH": 60.0,
                "AH": 0.8,
                "hour": 10,
                "day_of_week": 2,
                "month": 4,
                "is_weekend": 0
            }
        },
        {
            "name": "High Pollution",
            "data": {
                "PT08_S1CO": 1800.0,
                "NMHCGT": 200.0,
                "C6H6GT": 20.0,
                "PT08_S2NMHC": 1500.0,
                "NOxGT": 400.0,
                "PT08_S3NOx": 1200.0,
                "NO2GT": 200.0,
                "PT08_S4NO2": 2000.0,
                "PT08_S5O3": 1500.0,
                "T": 25.0,
                "RH": 40.0,
                "AH": 1.2,
                "hour": 18,
                "day_of_week": 4,
                "month": 7,
                "is_weekend": 0
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 60)
        
        # Test all three models
        for endpoint, model_name in [
            ("predict_model1", "XGBoost"),
            ("predict_model2", "Random Forest"),
            ("predict_model3", "Neural Network")
        ]:
            response = requests.post(
                f"{BASE_URL}/{endpoint}",
                json=scenario['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  {model_name:<20}: CO(GT) = {result['prediction']:.4f}")
            else:
                print(f"  {model_name:<20}: Error - {response.status_code}")
        
        print()

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("AIR QUALITY PREDICTION API - TEST CLIENT")
    print("=" * 80)
    print(f"\nTesting API at: {BASE_URL}")
    print(f"Time: {datetime.now()}\n")
    
    try:
        # Test health check
        test_health_check()
        
        # Test model info
        test_model_info()
        
        # Test individual predictions
        test_prediction("predict_model1", "XGBoost")
        test_prediction("predict_model2", "Random Forest")
        test_prediction("predict_model3", "Neural Network")
        
        # Test batch predictions
        test_batch_predictions()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  uv run uvicorn api.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
