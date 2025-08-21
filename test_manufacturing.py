import asyncio
import requests
import json

async def test_manufacturing_data():
    """Test complex queries on manufacturing data"""
    
    base_url = "http://localhost:8000"
    
    # Complex queries to test
    test_queries = [
        # Production Analysis
        "Compare the performance of Line A vs Line B",
        "Which shift (Morning vs Evening) produces better quality scores?",
        "Show me the production data for Line A on January 3rd, 2024",
        
        # Statistical Analysis
        "What is the average OEE percentage across all production lines?",
        "Find the correlation between Temperature and Quality Score",
        "Which production line has the lowest downtime?",
        
        # Time Series Analysis
        "Show me the trend of Units Produced over the 5-day period",
        "How does Quality Score change from January 1st to January 5th?",
        
        # Advanced Analytics
        "Calculate the total defects per production line and rank them",
        "What's the relationship between Humidity and Defects?",
        "Find the best performing shift for each production line",
        
        # Business Intelligence
        "Which day had the highest overall production efficiency?",
        "Recommend optimization strategies based on the data patterns",
        "Identify the top 3 factors affecting production quality"
    ]
    
    try:
        print("🏭 MANUFACTURING DATA ANALYSIS TESTING")
        print("="*80)
        
        # Create session
        print("1. Creating session...")
        response = requests.post(f"{base_url}/session/create", json={})
        if response.status_code != 200:
            print(f"❌ Failed to create session: {response.status_code}")
            print(response.text)
            return
        
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session created: {session_id}")
        
        # Upload manufacturing data
        print("\n2. Uploading manufacturing data...")
        with open("sample_manufacturing_data.csv", "rb") as f:
            files = {"file": ("sample_manufacturing_data.csv", f, "text/csv")}
            response = requests.post(
                f"{base_url}/session/{session_id}/upload",
                files=files
            )
        
        if response.status_code != 200:
            print(f"❌ Failed to upload data: {response.status_code}")
            print(response.text)
            return
        
        print("✅ Manufacturing data uploaded successfully")
        
        # Test each query
        print(f"\n🧪 TESTING {len(test_queries)} COMPLEX MANUFACTURING QUERIES")
        print("="*80)
        
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            print("-" * 80)
            
            chat_data = {
                "session_id": session_id,
                "query": query
            }
            
            try:
                response = requests.post(f"{base_url}/chat", json=chat_data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['response']
                    
                    print(f"✅ SUCCESS! Response length: {len(response_text)}")
                    
                    # Show first 300 characters
                    preview = response_text[:300].replace('\n', ' ')
                    print(f"📋 Preview: {preview}...")
                    
                    # Check for key indicators of good responses
                    indicators = {
                        "numerical_analysis": any(word in response_text.lower() for word in ['average', 'correlation', 'percentage', 'total', 'trend']),
                        "comparative_analysis": any(word in response_text.lower() for word in ['compare', 'vs', 'better', 'higher', 'lower']),
                        "data_insights": any(word in response_text.lower() for word in ['insight', 'pattern', 'recommendation', 'analysis']),
                        "specific_values": any(char.isdigit() for char in response_text[:500])  # Check for numbers in first 500 chars
                    }
                    
                    active_indicators = [k for k, v in indicators.items() if v]
                    if active_indicators:
                        print(f"🎯 Analysis types detected: {', '.join(active_indicators)}")
                    
                    successful_queries += 1
                    
                else:
                    print(f"❌ Query failed: {response.status_code}")
                    error_detail = response.json().get('detail', 'Unknown error') if response.text else 'No response'
                    print(f"   Error: {error_detail}")
                    
            except requests.exceptions.Timeout:
                print("⏱️ Query timed out (>30 seconds)")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Summary
        print(f"\n🎉 TESTING COMPLETED!")
        print("="*80)
        print(f"✅ Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"📊 Success rate: {(successful_queries/len(test_queries)*100):.1f}%")
        
        if successful_queries == len(test_queries):
            print("🚀 PERFECT! All complex manufacturing queries working!")
        elif successful_queries >= len(test_queries) * 0.8:
            print("🎯 EXCELLENT! Most queries working well!")
        elif successful_queries >= len(test_queries) * 0.6:
            print("✨ GOOD! System handling most queries!")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Some queries failing")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_manufacturing_data())
