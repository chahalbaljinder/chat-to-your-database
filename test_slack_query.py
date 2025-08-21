import asyncio
import requests
import json

async def test_slack_query():
    """Test the Slack query to see exactly what's being returned"""
    
    base_url = "http://localhost:8000"
    
    try:
        # Create a session
        print("1. Creating session...")
        session_request = {
            "user_preferences": {}
        }
        response = requests.post(f"{base_url}/session/create", json=session_request)
        if response.status_code != 200:
            print(f"Failed to create session: {response.status_code}")
            print(f"Response text: {response.text}")
            return
        
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session created: {session_id}")
        
        # Check if we have the sample dataset to upload
        import os
        dataset_path = "sample_data.csv"
        if os.path.exists(dataset_path):
            print(f"\n2. Uploading dataset: {dataset_path}")
            with open(dataset_path, 'rb') as f:
                files = {'file': f}
                upload_response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
                if upload_response.status_code == 200:
                    print("✅ Dataset uploaded successfully")
                else:
                    print(f"❌ Dataset upload failed: {upload_response.status_code}")
                    print(upload_response.text)
        else:
            print(f"\n2. Dataset file not found: {dataset_path}")
            print("Assuming dataset is already uploaded from previous session...")
        
        # Let's try the chat query directly
        print("\n3. Testing Slack query...")
        
        chat_data = {
            "session_id": session_id,
            "message": "Show me the data for Slack"
        }
        
        response = requests.post(f"{base_url}/chat", json=chat_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"Response length: {len(result['response'])}")
            print("\n--- RESPONSE ---")
            print(result['response'][:1000])  # First 1000 chars
            if len(result['response']) > 1000:
                print("... (truncated)")
                print(f"\nFull length: {len(result['response'])} characters")
            
            # Check if it's showing all data or just Slack
            response_lower = result['response'].lower()
            if 'discord' in response_lower or 'teams' in response_lower or 'zoom' in response_lower:
                print("\n❌ ISSUE: Response contains data from other platforms!")
                print("The system is showing ALL data instead of filtering for Slack only.")
            else:
                print("\n✅ GOOD: Response appears to be filtered for Slack only.")
                
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_slack_query())
