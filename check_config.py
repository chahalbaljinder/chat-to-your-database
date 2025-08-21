"""
Configuration validator for Agentic Data Chat System
"""
import os
import sys
from pathlib import Path

def check_configuration():
    """Check if the system is properly configured"""
    
    print("🔧 Agentic Data Chat System - Configuration Check")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Check Python version
    print("1. Checking Python version...")
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"Python {python_version.major}.{python_version.minor} is too old. Requires Python 3.8+")
    else:
        print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check .env file
    print("\n2. Checking .env file...")
    if not os.path.exists(".env"):
        issues.append(".env file not found. Run setup.bat to create one.")
    else:
        print("   ✅ .env file exists")
        
        # Check API key
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key or api_key == "your_api_key_here":
                issues.append("GOOGLE_API_KEY not set in .env file")
            elif len(api_key) < 20:
                warnings.append("GOOGLE_API_KEY seems too short")
            else:
                print(f"   ✅ API key configured ({len(api_key)} characters)")
        except ImportError:
            warnings.append("python-dotenv not installed - can't validate .env")
    
    # Check required directories
    print("\n3. Checking directory structure...")
    required_dirs = ["agents", "config", "utils", "temp", "logs"]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/")
        else:
            warnings.append(f"Directory '{directory}' missing (will be created automatically)")
    
    # Check core files
    print("\n4. Checking core files...")
    core_files = [
        "main.py",
        "orchestrator.py", 
        "context_manager.py",
        "requirements.txt",
        "config/settings.py",
        "config/agent_prompts.py",
        "utils/session_utils.py",
        "utils/data_loader.py",
        "agents/base_agent.py"
    ]
    
    missing_files = []
    for file_path in core_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
    
    if missing_files:
        issues.append(f"Missing files: {', '.join(missing_files)}")
    
    # Check dependencies (if possible)
    print("\n5. Checking dependencies...")
    try:
        import google.generativeai
        print("   ✅ google-generativeai")
    except ImportError:
        warnings.append("google-generativeai not installed")
    
    try:
        import pandas
        print("   ✅ pandas")
    except ImportError:
        warnings.append("pandas not installed")
    
    try:
        import fastapi
        print("   ✅ fastapi")
    except ImportError:
        warnings.append("fastapi not installed")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Configuration Summary")
    print("=" * 60)
    
    if issues:
        print("❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
        print()
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    
    if not issues and not warnings:
        print("✅ All checks passed! System is ready to run.")
    elif not issues:
        print("✅ System should work, but check warnings above.")
    else:
        print("❌ Please fix critical issues before running the system.")
    
    print("\n📚 Next Steps:")
    if issues:
        print("   1. Fix critical issues listed above")
        if ".env" in str(issues):
            print("   2. Run setup.bat to create .env file")
            print("   3. Add your Google API key to .env file")
        if "not installed" in str(warnings):
            print("   4. Run: pip install -r requirements.txt")
    else:
        print("   1. Run: python main.py (to start the server)")
        print("   2. Open: http://localhost:8000/docs (for API docs)")
        print("   3. Upload data and start chatting!")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = check_configuration()
    sys.exit(0 if success else 1)
