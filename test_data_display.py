"""
Test script to verify the improved data display functionality
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_data_display():
    """Test the new data display functionality"""
    
    print("🔍 Testing Data Display Functionality\n")
    
    # Step 1: Create session
    print("1. Creating session...")
    session_response = requests.post(f"{BASE_URL}/session/create")
    
    if session_response.status_code != 200:
        print(f"❌ Failed to create session: {session_response.status_code}")
        return False
    
    session_data = session_response.json()
    session_id = session_data["session_id"]
    print(f"✅ Session created: {session_id}")
    
    # Step 2: Upload test data
    print("\n2. Uploading test dataset...")
    
    # Use the existing test data file
    test_file_path = r"C:\Users\balli\Desktop\Projects\chat-to-your-database\test_data.csv"
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            upload_response = requests.post(
                f"{BASE_URL}/session/{session_id}/upload",
                files=files
            )
        
        if upload_response.status_code != 200:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"Response: {upload_response.text}")
            return False
        
        print("✅ Dataset uploaded successfully")
        
    except FileNotFoundError:
        print(f"❌ Test data file not found: {test_file_path}")
        return False
    
    # Step 3: Test data display queries
    test_queries = [
        "show the data of slack",
        "display Microsoft Teams information",
        "show me zoom data",
        "get discord details"
    ]
    
    print(f"\n3. Testing data display queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        
        chat_data = {
            "session_id": session_id,
            "query": query
        }
        
        chat_response = requests.post(
            f"{BASE_URL}/chat",
            json=chat_data,
            headers={"Content-Type": "application/json"}
        )
        
        if chat_response.status_code != 200:
            print(f"❌ Chat request failed: {chat_response.status_code}")
            print(f"Response: {chat_response.text}")
            continue
        
        try:
            response_data = chat_response.json()
            
            if response_data.get("success"):
                content = response_data["response"]["content"]
                response_type = response_data["response"].get("response_type", "unknown")
                
                print(f"✅ Response Type: {response_type}")
                print(f"📋 Content Preview: {content[:200]}...")
                
                # Check if it's showing actual data vs just guidance
                if "Row 1:" in content or "| Platform |" in content or "Found" in content:
                    print("🎯 SUCCESS: Showing actual data results!")
                elif "strategic guide" in content.lower() or "analysis approaches" in content.lower():
                    print("⚠️  WARNING: Still showing guidance instead of data")
                else:
                    print("ℹ️  Response type unclear")
                    
            else:
                print(f"❌ Chat failed: {response_data.get('error', 'Unknown error')}")
                
        except json.JSONDecodeError:
            print("❌ Failed to parse response JSON")
        
        time.sleep(1)  # Brief pause between requests
    
    print(f"\n🎉 Data Display Testing Complete!")
    return True

if __name__ == "__main__":
    test_data_display()
