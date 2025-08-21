import requests
import json

def test_quick():
    """Quick test to see what happens with Slack query"""
    try:
        # Just test with an existing session ID we know works
        base_url = "http://localhost:8000"
        
        # Try session creation first
        print("Testing session creation...")
        session_request = {"user_preferences": {}}
        response = requests.post(f"{base_url}/session/create", json=session_request)
        
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data['session_id']
            print(f"✅ Session: {session_id}")
            
            # Upload sample data
            try:
                with open("sample_data.csv", 'rb') as f:
                    files = {'file': f}
                    upload_response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
                    if upload_response.status_code == 200:
                        print("✅ Data uploaded")
                        
                        # Test the Slack query
                        chat_data = {
                            "session_id": session_id,
                            "query": "Show me the data for Slack"
                        }
                        
                        response = requests.post(f"{base_url}/chat", json=chat_data)
                        if response.status_code == 200:
                            result = response.json()
                            print(f"✅ Chat response received ({len(result['response'])} chars)")
                            
                            # Check what's in the response
                            response_text = result['response'].lower()
                            platforms_mentioned = []
                            
                            test_platforms = ['slack', 'discord', 'teams', 'zoom', 'google', 'telegram', 'whatsapp', 'signal']
                            for platform in test_platforms:
                                if platform in response_text:
                                    platforms_mentioned.append(platform)
                            
                            print(f"Platforms mentioned: {platforms_mentioned}")
                            
                            if len(platforms_mentioned) == 1 and 'slack' in platforms_mentioned:
                                print("✅ GOOD: Only Slack mentioned")
                            elif 'slack' in platforms_mentioned and len(platforms_mentioned) > 1:
                                print("❌ ISSUE: Multiple platforms mentioned, not filtering properly")
                            else:
                                print("❌ ISSUE: Unexpected result")
                                
                            # Show first 300 chars
                            print("\nFirst 300 chars of response:")
                            print("-" * 50)
                            print(result['response'][:300])
                            print("-" * 50)
                        else:
                            print(f"❌ Chat failed: {response.status_code}")
                    else:
                        print(f"❌ Upload failed: {upload_response.status_code}")
            except FileNotFoundError:
                print("❌ sample_data.csv not found")
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_quick()
