"""
Test script for Agentic Data Chat System
"""
import asyncio
import json
import os
from datetime import datetime

# Import system components
from orchestrator import Orchestrator
from context_manager import ContextManager
from utils.session_utils import SessionManager
from utils.data_loader import DataLoader
from config.settings import SETTINGS

async def test_system():
    """Test the core functionality of the system"""
    
    print("🚀 Testing Agentic Data Chat System")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing system components...")
    orchestrator = Orchestrator()
    context_manager = ContextManager()
    session_manager = SessionManager()
    data_loader = DataLoader()
    
    print("✅ Components initialized successfully")
    
    # Test session creation
    print("\n2. Testing session management...")
    session_context = session_manager.create_session({
        "test_mode": True,
        "detail_level": "medium"
    })
    print(f"✅ Session created: {session_context.session_id}")
    
    # Test context manager
    print("\n3. Testing context manager...")
    context_result = await context_manager.process(
        query="Test context management",
        context=session_context,
        operation="manage"
    )
    print(f"✅ Context manager response: {context_result.get('success', False)}")
    
    # Test orchestrator without data first
    print("\n4. Testing orchestrator (without data)...")
    test_queries = [
        "Hello, can you help me analyze data?",
        "What can you do for me?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = await orchestrator.process_query(
                query=query,
                context=session_context
            )
            if result.get("success"):
                response = result.get("response")
                print(f"✅ Response: {response.content[:100]}..." if response else "✅ Success (no content)")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Test individual agents
    print("\n5. Testing individual agents...")
    
    # Test with mock data for agents that need it
    mock_data = {
        "columns": ["sales", "marketing", "region"],
        "dtypes": {"sales": "float64", "marketing": "float64", "region": "object"},
        "shape": (1000, 3),
        "sample_data": [
            {"sales": 100, "marketing": 50, "region": "North"},
            {"sales": 120, "marketing": 60, "region": "South"}
        ]
    }
    
    # Update session with mock data info
    session_context.dataset_info = mock_data
    
    agent_tests = [
        ("data_understanding", "Analyze the structure of this dataset"),
        ("query_processing", "Show me sales by region"),
        ("data_analysis", "Calculate correlation between sales and marketing"),
        ("visualization", "Create a bar chart of sales by region"),
        ("insight_generation", "What insights can you provide about this data?")
    ]
    
    for agent_type, query in agent_tests:
        print(f"\nTesting {agent_type} agent...")
        try:
            result = await orchestrator.process_query(
                query=f"[TEST_{agent_type.upper()}] {query}",
                context=session_context
            )
            if result.get("success"):
                print(f"✅ {agent_type} agent working")
            else:
                print(f"❌ {agent_type} agent error: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ {agent_type} agent exception: {str(e)}")
    
    # Test session cleanup
    print("\n6. Testing session cleanup...")
    cleanup_result = context_manager.cleanup_expired_sessions()
    print(f"✅ Cleanup completed: {cleanup_result}")
    
    # Test compression
    print("\n7. Testing context compression...")
    # Add some conversation history first
    from utils.session_utils import ConversationTurn
    
    for i in range(10):
        turn = ConversationTurn(
            user_query=f"Test query {i}",
            assistant_response=f"Test response {i}",
            timestamp=datetime.now(),
            agent_responses=[],
            metadata={}
        )
        session_context.add_conversation_turn(turn)
    
    compress_result = await context_manager.process(
        query="compress",
        context=session_context,
        operation="compress"
    )
    print(f"✅ Compression test: {compress_result.get('success', False)}")
    
    print("\n" + "=" * 50)
    print("🎉 System testing completed!")
    print("\nSystem Status:")
    print(f"  Model: {SETTINGS.MODEL_NAME}")
    print(f"  Session timeout: {SETTINGS.SESSION_TIMEOUT_MINUTES} minutes")
    print(f"  Max file size: {SETTINGS.MAX_FILE_SIZE_MB} MB")
    print(f"  Temperature: {SETTINGS.TEMPERATURE}")
    
    return True

async def test_api_endpoints():
    """Test API endpoints (requires running server)"""
    
    print("\n🌐 Testing API Endpoints")
    print("=" * 30)
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health check
        print("Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test session creation
        print("\nTesting session creation...")
        response = requests.post(
            f"{base_url}/session/create",
            json={"user_preferences": {"test_mode": True}},
            timeout=10
        )
        
        if response.status_code == 200:
            session_data = response.json()
            print("✅ Session creation passed")
            session_id = session_data.get("session_id")
            print(f"   Session ID: {session_id}")
            
            # Test chat endpoint
            print("\nTesting chat endpoint...")
            chat_response = requests.post(
                f"{base_url}/chat",
                json={
                    "session_id": session_id,
                    "query": "Hello, what can you help me with?"
                },
                timeout=30
            )
            
            if chat_response.status_code == 200:
                print("✅ Chat endpoint passed")
                chat_data = chat_response.json()
                print(f"   Response: {chat_data.get('response', '')[:100]}...")
            else:
                print(f"❌ Chat endpoint failed: {chat_response.status_code}")
                print(f"   Error: {chat_response.text}")
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ API server not running")
        print("   Start the server with: python main.py")
    except Exception as e:
        print(f"❌ API test error: {str(e)}")

def test_configuration():
    """Test system configuration"""
    
    print("\n⚙️ Testing Configuration")
    print("=" * 25)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ Google API key found")
        print(f"   Key length: {len(api_key)} characters")
    else:
        print("❌ Google API key not found")
        print("   Please set GOOGLE_API_KEY in .env file")
    
    # Check settings
    print(f"\nConfiguration:")
    print(f"  Model: {SETTINGS.MODEL_NAME}")
    print(f"  Temperature: {SETTINGS.TEMPERATURE}")
    print(f"  Max tokens: {SETTINGS.MAX_TOKENS}")
    print(f"  Session timeout: {SETTINGS.SESSION_TIMEOUT_MINUTES} min")
    print(f"  Max file size: {SETTINGS.MAX_FILE_SIZE_MB} MB")
    
    # Check required directories
    directories = ["temp", "logs"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Directory '{directory}' exists")
        else:
            print(f"⚠️ Directory '{directory}' missing (will be created)")
            os.makedirs(directory, exist_ok=True)
            print(f"   Created '{directory}' directory")

if __name__ == "__main__":
    print("🧪 Agentic Data Chat System Test Suite")
    print("=" * 60)
    
    # Test configuration first
    test_configuration()
    
    # Test core system
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {str(e)}")
    
    # Test API endpoints if available
    print("\n" + "=" * 60)
    try:
        asyncio.run(test_api_endpoints())
    except Exception as e:
        print(f"API testing error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🏁 Testing completed!")
    print("\nNext steps:")
    print("1. Fix any errors shown above")
    print("2. Run: python main.py (to start the server)")
    print("3. Visit: http://localhost:8000/docs (for API documentation)")
    print("4. Upload data and start chatting with your data!")
