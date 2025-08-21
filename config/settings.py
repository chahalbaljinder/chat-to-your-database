"""
Configuration settings for the Agentic Data Chat system
"""
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash"
    MODEL_NAME = GEMINI_MODEL  # Alias for compatibility
    
    # Session Management
    SESSION_STORAGE_PATH = os.getenv("SESSION_STORAGE_PATH", "./sessions")
    DATA_CACHE_PATH = os.getenv("DATA_CACHE_PATH", "./data_cache")
    SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "2"))
    SESSION_TIMEOUT_MINUTES = SESSION_TIMEOUT_HOURS * 60  # Compatibility alias
    MAX_HISTORY_TURNS = 20
    CONTEXT_COMPRESSION = True
    AUTO_SAVE_SESSION = True
    CONTEXT_RELEVANCE_THRESHOLD = 0.7
    
    # Agent Configuration
    AGENT_CONFIG = {
        "temperature": 0.1,
        "max_tokens": 2048,
        "context_window": 10,
        "enable_memory": True,
        "parallel_processing": True,
        "timeout_seconds": 30
    }
    
    # Compatibility aliases for direct access
    TEMPERATURE = AGENT_CONFIG["temperature"]
    MAX_TOKENS = AGENT_CONFIG["max_tokens"]
    
    # Data Processing
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    MAX_ROWS = 1000000
    CACHE_ANALYSIS_RESULTS = True
    AUTO_DATA_PROFILING = True
    SUPPORTED_FORMATS = ["csv", "xlsx", "json", "parquet"]
    
    # Logging
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

# Create settings instance
SETTINGS = Config()
