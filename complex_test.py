import asyncio
import requests
import json
import time

class ComplexQueryTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        
    def create_session_and_upload(self):
        """Create session and upload dataset"""
        print("🚀 Setting up test environment...")
        
        # Create session
        response = requests.post(f"{self.base_url}/session/create", json={})
        if response.status_code != 200:
            print(f"❌ Failed to create session: {response.status_code}")
            return False
            
        self.session_id = response.json()['session_id']
        print(f"✅ Session created: {self.session_id}")
        
        # Upload dataset
        with open('sample_data.csv', 'rb') as f:
            files = {'file': ('sample_data.csv', f, 'text/csv')}
            response = requests.post(
                f"{self.base_url}/session/{self.session_id}/upload", 
                files=files
            )
            
        if response.status_code == 200:
            print("✅ Dataset uploaded successfully")
            return True
        else:
            print(f"❌ Failed to upload dataset: {response.status_code}")
            return False
    
    def test_query(self, query, expected_keywords=None, unexpected_keywords=None):
        """Test a complex query and analyze the response"""
        print(f"\n🔍 Testing: '{query}'")
        print("-" * 80)
        
        chat_data = {
            "session_id": self.session_id,
            "query": query
        }
        
        response = requests.post(f"{self.base_url}/chat", json=chat_data)
        
        if response.status_code != 200:
            print(f"❌ Query failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        result = response.json()
        response_text = result['response'].lower()
        response_length = len(result['response'])
        
        print(f"✅ SUCCESS! Response length: {response_length}")
        
        # Check for expected keywords
        if expected_keywords:
            found_expected = []
            missing_expected = []
            for keyword in expected_keywords:
                if keyword.lower() in response_text:
                    found_expected.append(keyword)
                else:
                    missing_expected.append(keyword)
                    
            if found_expected:
                print(f"✅ Found expected keywords: {found_expected}")
            if missing_expected:
                print(f"⚠️  Missing expected keywords: {missing_expected}")
        
        # Check for unexpected keywords
        if unexpected_keywords:
            found_unexpected = []
            for keyword in unexpected_keywords:
                if keyword.lower() in response_text:
                    found_unexpected.append(keyword)
                    
            if found_unexpected:
                print(f"❌ Found unexpected keywords: {found_unexpected}")
            else:
                print(f"✅ No unexpected keywords found")
        
        # Show response preview
        print(f"\n📋 Response Preview (first 300 chars):")
        print("=" * 60)
        print(result['response'][:300])
        if len(result['response']) > 300:
            print("... (truncated)")
        print("=" * 60)
        
        return True

def main():
    tester = ComplexQueryTester()
    
    # Setup
    if not tester.create_session_and_upload():
        return
        
    # Wait for server to process
    time.sleep(2)
    
    print("\n🧪 STARTING COMPLEX QUERY TESTS")
    print("=" * 80)
    
    # Test 1: Comparative Analysis
    tester.test_query(
        "Compare Slack vs Discord in terms of features and limitations",
        expected_keywords=["Slack", "Discord", "features", "limitations"],
        unexpected_keywords=["Teams", "Zoom", "WhatsApp"]
    )
    
    # Test 2: Multi-criteria Filtering
    tester.test_query(
        "Show me all platforms that are best for small teams and have unlimited users",
        expected_keywords=["small teams", "unlimited"],
        unexpected_keywords=["limited", "enterprise"]
    )
    
    # Test 3: Feature-based Query
    tester.test_query(
        "Which platforms support file sharing and have mobile apps?",
        expected_keywords=["file sharing", "mobile"],
        unexpected_keywords=[]
    )
    
    # Test 4: Limitation Analysis
    tester.test_query(
        "What are the common limitations across all communication platforms?",
        expected_keywords=["limitations", "common"],
        unexpected_keywords=[]
    )
    
    # Test 5: Recommendation Query
    tester.test_query(
        "Recommend the best platform for a startup with 20 employees who need project management features",
        expected_keywords=["startup", "20", "project management", "recommend"],
        unexpected_keywords=[]
    )
    
    # Test 6: Data Summary Query
    tester.test_query(
        "Give me a summary of all platforms including their key strengths and weaknesses",
        expected_keywords=["summary", "strengths", "weaknesses"],
        unexpected_keywords=[]
    )
    
    # Test 7: Specific Business Case
    tester.test_query(
        "I need a platform for remote team collaboration with good integration capabilities and no user limits",
        expected_keywords=["remote", "collaboration", "integration", "no user limits"],
        unexpected_keywords=[]
    )
    
    # Test 8: Cost Analysis
    tester.test_query(
        "Show me only the free platforms and their main features",
        expected_keywords=["free", "features"],
        unexpected_keywords=["paid", "premium"]
    )
    
    # Test 9: Technical Requirements
    tester.test_query(
        "Which platforms offer end-to-end encryption and can be self-hosted?",
        expected_keywords=["encryption", "self-hosted"],
        unexpected_keywords=[]
    )
    
    # Test 10: Complex Conditional Query
    tester.test_query(
        "If I have a team of 50 people and need advanced features but want to stay on the free tier, what are my options?",
        expected_keywords=["50", "advanced features", "free tier", "options"],
        unexpected_keywords=[]
    )
    
    # Test 11: Storage Analysis (New - leveraging full dataset)
    tester.test_query(
        "Which platforms have storage limitations and what are the specific limits?",
        expected_keywords=["storage", "limitations", "5GB", "100GB"],
        unexpected_keywords=[]
    )
    
    # Test 12: User Limit Comparison (New)
    tester.test_query(
        "Compare platforms with user limits vs unlimited users - which ones restrict team size?",
        expected_keywords=["user limits", "unlimited", "team size", "Rocket.Chat", "Bitrix24"],
        unexpected_keywords=[]
    )
    
    # Test 13: Meeting Duration Analysis (New)
    tester.test_query(
        "Which platforms have meeting time restrictions and what are the limits?",
        expected_keywords=["meeting", "time", "40-minute", "60-minute", "Zoom", "Microsoft Teams"],
        unexpected_keywords=[]
    )
    
    # Test 14: Open Source & Privacy Focus (New)
    tester.test_query(
        "Show me platforms that prioritize privacy, offer self-hosting, or are open-source",
        expected_keywords=["privacy", "self-hosted", "open-source", "Element", "Rocket.Chat"],
        unexpected_keywords=[]
    )
    
    # Test 15: Integration Capabilities (New)
    tester.test_query(
        "Which platforms offer the best integration capabilities and what are the limitations?",
        expected_keywords=["integration", "Google Workspace", "Office", "10 app integrations"],
        unexpected_keywords=[]
    )
    
    print("\n🎉 COMPLEX QUERY TESTING COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
