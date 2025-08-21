import asyncio
import requests
import json

async def test_comparative_fix():
    """Test the comparative analysis fix"""
    
    base_url = "http://localhost:8000"
    
    try:
        # Create session
        print("1. Creating session...")
        response = requests.post(f"{base_url}/session/create", json={})
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session: {session_id}")
        
        # Upload data
        print("2. Uploading data...")
        with open("sample_data.csv", "rb") as f:
            files = {"file": ("sample_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
        
        if response.status_code == 200:
            print("✅ Data uploaded")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return
        
        # Test comparative query
        print("3. Testing comparative query...")
        chat_data = {
            "session_id": session_id,
            "query": "Compare Slack vs Discord in terms of features and limitations"
        }
        
        response = requests.post(f"{base_url}/chat", json=chat_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Query processed")
            print(f"Response length: {len(result['response'])}")
            print("\n--- FIRST 500 CHARACTERS ---")
            print(result['response'][:500])
            print("...")
            
        else:
            print(f"❌ Query failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_comparative_fix())
