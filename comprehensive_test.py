#!/usr/bin/env python3

import requests
import json
import pandas as pd
import os
import time

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_session_creation():
    """Test session creation"""
    print("\n🔍 Testing session creation...")
    try:
        response = requests.post("http://localhost:8000/session/create")
        if response.status_code == 200:
            data = response.json()
            session_id = data.get('session_id')
            print(f"✅ Session created: {session_id}")
            return session_id
        else:
            print(f"❌ Session creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def test_file_upload(session_id):
    """Test file upload functionality"""
    print(f"\n🔍 Testing file upload for session {session_id}...")
    
    # Create test data with various data types
    test_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Adams'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000.50, 60000.75, 70000.25, 55000.00, 65000.90],
        'department': ['Engineering', 'HR', 'Finance', 'Marketing', 'Engineering'],
        'start_date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-01', '2019-11-30'],
        'active': [True, True, False, True, True]
    }
    
    df = pd.DataFrame(test_data)
    test_file = 'test_employees.csv'
    df.to_csv(test_file, index=False)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            response = requests.post(
                f"http://localhost:8000/session/{session_id}/upload", 
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ File upload successful!")
            print(f"   Dataset shape: {data.get('dataset_info', {}).get('shape')}")
            print(f"   Columns: {len(data.get('dataset_info', {}).get('columns', []))}")
            return True
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_session_status(session_id):
    """Test session status endpoint"""
    print(f"\n🔍 Testing session status for {session_id}...")
    try:
        response = requests.get(f"http://localhost:8000/session/{session_id}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Session status retrieved successfully")
            print(f"   Active: {data.get('active')}")
            print(f"   Datasets: {len(data.get('datasets', []))}")
            return True
        else:
            print(f"❌ Session status failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Session status error: {e}")
        return False

def test_chat_functionality(session_id):
    """Test chat endpoint"""
    print(f"\n🔍 Testing chat functionality for {session_id}...")
    
    # Test queries
    test_queries = [
        "What columns are in my dataset?",
        "How many employees are there?",
        "Show me basic statistics about the data"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n   Query {i+1}: {query}")
        try:
            chat_data = {
                "session_id": session_id,
                "message": query,
                "include_context": True
            }
            
            response = requests.post(
                "http://localhost:8000/chat", 
                json=chat_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response received (length: {len(data.get('response', ''))} chars)")
                # Print first 100 characters of response
                response_text = data.get('response', '')
                if len(response_text) > 100:
                    print(f"   Preview: {response_text[:100]}...")
                else:
                    print(f"   Response: {response_text}")
            else:
                print(f"   ❌ Chat failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Chat error: {e}")
            return False
    
    print("✅ Chat functionality tests completed")
    return True

def main():
    """Run comprehensive API tests"""
    print("🚀 Starting Agentic Data Chat API Tests")
    print("="*50)
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed. Server may not be running.")
        return
    
    # Test session creation
    session_id = test_session_creation()
    if not session_id:
        print("❌ Cannot proceed without a valid session.")
        return
    
    # Test file upload
    if not test_file_upload(session_id):
        print("❌ File upload failed.")
        return
    
    # Test session status
    if not test_session_status(session_id):
        print("❌ Session status check failed.")
        return
    
    # Test chat functionality
    if not test_chat_functionality(session_id):
        print("❌ Chat functionality failed.")
        return
    
    print("\n" + "="*50)
    print("🎉 All tests passed! The Agentic Data Chat API is working correctly.")
    print(f"   ✅ Health check")
    print(f"   ✅ Session management") 
    print(f"   ✅ File upload with numpy serialization")
    print(f"   ✅ Data processing")
    print(f"   ✅ Chat functionality")
    print("="*50)

if __name__ == "__main__":
    main()
