import requests
import json

def test_complex_query():
    """Test a complex query that was previously failing"""
    
    base_url = "http://localhost:8000"
    
    try:
        # Create session
        print("Creating session...")
        response = requests.post(f"{base_url}/session/create", json={})
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session: {session_id}")
        
        # Upload data
        print("Uploading data...")
        with open('sample_data.csv', 'rb') as f:
            files = {'file': ('sample_data.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
        
        if response.status_code == 200:
            print("✅ Data uploaded")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return
        
        # Test a complex query that was previously failing
        print("Testing complex query...")
        chat_data = {
            "session_id": session_id,
            "query": "Compare Slack vs Discord in terms of features and limitations"
        }
        
        response = requests.post(f"{base_url}/chat", json=chat_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Complex query worked")
            print(f"Response length: {len(result['response'])}")
            print("\n=== RESPONSE PREVIEW ===")
            print(result['response'][:500])
            print("..." if len(result['response']) > 500 else "")
            print("========================")
            
        else:
            print(f"❌ Query failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error: {error_detail}")
            except:
                print(f"Error text: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_complex_query()
