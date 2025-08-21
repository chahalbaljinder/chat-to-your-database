#!/usr/bin/env python3

import requests
import json

# Test script for file upload functionality

def test_session_creation():
    """Test session creation"""
    url = "http://localhost:8000/session/create"
    response = requests.post(url)
    print(f"Session creation status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Session ID: {data.get('session_id')}")
        return data.get('session_id')
    else:
        print(f"Error: {response.text}")
        return None

def test_file_upload(session_id):
    """Test file upload"""
    # Create a simple test CSV file
    import pandas as pd
    import os
    
    # Create test data
    test_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Tokyo', 'Paris', 'Berlin'],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'Finance', 'Marketing', 'IT']
    }
    
    df = pd.DataFrame(test_data)
    test_file = 'test_employees.csv'
    df.to_csv(test_file, index=False)
    
    try:
        # Upload file
        url = f"http://localhost:8000/session/{session_id}/upload"
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            response = requests.post(url, files=files)
        
        print(f"File upload status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Upload successful!")
            print(f"Dataset info: {json.dumps(data.get('dataset_info', {}), indent=2)}")
            return True
        else:
            print(f"Upload failed: {response.text}")
            return False
            
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    print("Testing Agentic Data Chat API")
    print("="*40)
    
    # Test session creation
    session_id = test_session_creation()
    if not session_id:
        print("Failed to create session. Exiting.")
        return
    
    print("\n" + "="*40)
    
    # Test file upload
    upload_success = test_file_upload(session_id)
    if upload_success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Upload test failed!")

if __name__ == "__main__":
    main()
