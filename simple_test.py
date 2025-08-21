import requests
import json
import time

def simple_test():
    """Simple test to check if Slack query returns all data or filtered data"""
    try:
        base_url = "http://localhost:8000"
        
        # Use a known working session ID from server logs
        session_id = "98153af2-64e5-4137-bb10-ffad6d5acca8"  # From previous successful upload
        
        # Or create a new one
        print("Creating session...")
        session_request = {"user_preferences": {}}
        response = requests.post(f"{base_url}/session/create", json=session_request)
        
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data['session_id']
            print(f"Session: {session_id}")
            
            # Upload data
            with open("sample_data.csv", 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
                
            if response.status_code == 200:
                print("Data uploaded successfully")
                
                # Wait a moment
                time.sleep(1)
                
                # Test the chat
                chat_data = {
                    "session_id": session_id,
                    "query": "Show me the data for Slack"
                }
                
                print("Sending query...")
                response = requests.post(f"{base_url}/chat", json=chat_data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"\n✅ SUCCESS! Response length: {len(result['response'])}")
                    
                    # Check response content
                    response_text = result['response']
                    print("\n" + "="*60)
                    print("FIRST 500 CHARACTERS:")
                    print("="*60)
                    print(response_text[:500])
                    print("="*60)
                    
                    # Check if it mentions multiple platforms (indicating it's showing all data)
                    platforms = ['discord', 'teams', 'zoom', 'google', 'telegram', 'whatsapp', 'signal', 'matrix']
                    found_platforms = [p for p in platforms if p.lower() in response_text.lower()]
                    
                    if found_platforms:
                        print(f"\n❌ ISSUE CONFIRMED: Response mentions other platforms: {found_platforms}")
                        print("This indicates the system is showing ALL data instead of filtering for Slack only.")
                    else:
                        print(f"\n✅ GOOD: Only mentions Slack, not other platforms")
                        
                else:
                    print(f"❌ Chat request failed: {response.status_code}")
                    print(response.text)
            else:
                print(f"❌ Upload failed: {response.status_code}")
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_test()
