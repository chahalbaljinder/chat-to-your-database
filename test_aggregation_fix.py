import asyncio
import requests
import json

async def test_aggregation_queries():
    """Test the new aggregation analysis functionality"""
    
    base_url = "http://localhost:8000"
    
    # Aggregation test queries
    test_queries = [
        "Which day had the highest overall production efficiency?",
        "Which production line has the lowest downtime?",
        "What is the total defects per production line?", 
        "Which shift produces the best quality scores?",
        "What is the average OEE percentage across all production lines?"
    ]
    
    try:
        print("🔧 TESTING AGGREGATION ANALYSIS FIXES")
        print("="*70)
        
        # Create session
        print("1. Creating session...")
        response = requests.post(f"{base_url}/session/create", json={})
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session: {session_id}")
        
        # Upload data
        print("2. Uploading manufacturing data...")
        with open("sample_manufacturing_data.csv", "rb") as f:
            files = {"file": ("sample_manufacturing_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/session/{session_id}/upload", files=files)
        print("✅ Data uploaded")
        
        # Test queries
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{query}'")
            print("-" * 60)
            
            chat_data = {"session_id": session_id, "query": query}
            response = requests.post(f"{base_url}/chat", json=chat_data, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['response']
                
                print(f"✅ SUCCESS! Length: {len(response_text)}")
                
                # Check if it's giving specific answers vs general dataset overview
                if "Dataset Overview" in response_text:
                    print("❌ STILL SHOWING DATASET OVERVIEW")
                elif any(word in response_text for word in ["Answer:", "Result:", "Winner:", "highest", "lowest", "best"]):
                    print("✅ GIVING SPECIFIC ANSWER!")
                else:
                    print("⚠️  UNCLEAR - check manually")
                
                # Show preview
                preview = response_text[:200].replace('\n', ' ')
                print(f"📋 Preview: {preview}...")
                
            else:
                print(f"❌ FAILED: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_aggregation_queries())
