# Agentic Data Chat & Visualization System - Complete Instructions

## Overview
This system provides a conversational AI interface to interact with data using natural language, powered by Google's Gemini 2.0 Flash LLM. The architecture uses specialized agents that maintain context across chat sessions for seamless data exploration and visualization.

## System Architecture Overview

### Core Principles
- **Multi-Agent Architecture**: Specialized agents handle different aspects of data interaction
- **Conversational Context**: Maintains chat history and context across sessions
- **Session Management**: Automatic context refresh for new sessions
- **Natural Language Processing**: Understands complex data queries in plain English
- **Interactive Visualizations**: Generates appropriate charts and dashboards
- **Contextual Insights**: Provides business intelligence based on conversation flow

## Agent Hierarchy

### 1. **Orchestrator Agent** (Master Controller)
- **Purpose**: Central coordinator managing all specialized agents
- **Responsibilities**: 
  - Route queries to appropriate agents
  - Maintain session context
  - Coordinate multi-agent workflows
  - Manage conversation flow
  - Handle context refresh for new sessions

### 2. **Context Manager** (Session Handler)
- **Purpose**: Manages conversational context and session state
- **Responsibilities**:
  - Create and manage chat sessions
  - Store conversation history
  - Handle context refresh
  - Maintain user preferences
  - Cache analysis results

### 3. **Specialized Agents**

#### Data Understanding Agent
- **Role**: Analyzes and understands data structure
- **Context Awareness**: Remembers previous data explorations
- **Functions**:
  - Schema analysis and data profiling
  - Data quality assessment
  - Column type inference and validation
  - Missing value detection and handling

#### Query Processing Agent  
- **Role**: Interprets natural language queries with context
- **Context Awareness**: References previous queries and results
- **Functions**:
  - Intent classification with conversation history
  - Entity extraction and reference resolution
  - Query complexity assessment
  - SQL/Pandas code generation with context

#### Data Analysis Agent
- **Role**: Performs statistical analysis building on previous work
- **Context Awareness**: Builds on previous analyses and insights
- **Functions**:
  - Descriptive statistics and summaries
  - Correlation and relationship analysis
  - Trend identification and forecasting
  - Comparative analysis across conversation

#### Visualization Agent
- **Role**: Creates visualizations considering conversation flow
- **Context Awareness**: Suggests related visualizations based on history
- **Functions**:
  - Context-aware chart type selection
  - Interactive visualization creation
  - Multi-panel dashboards
  - Progressive visualization building

#### Insight Generation Agent
- **Role**: Generates insights building on conversation context
- **Context Awareness**: Synthesizes insights across entire session
- **Functions**:
  - Pattern interpretation with historical context
  - Recommendation generation
  - Contextual business intelligence
  - Conversation-aware insights

## Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install google-generativeai pandas numpy matplotlib seaborn plotly streamlit sqlite3 asyncio

# Optional for advanced features
pip install scikit-learn scipy statsmodels dash
```

### Environment Configuration
```bash
# Set your Google AI API key
export GOOGLE_API_KEY="your_api_key_here"

# Optional: Set custom storage paths
export SESSION_STORAGE_PATH="./chat_sessions"
export DATA_CACHE_PATH="./data_cache"
```

### Project Structure
```
agentic_data_chat/
├── main.py                     # Main application entry point
├── orchestrator.py             # Master orchestrator
├── context_manager.py          # Session and context management
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class with context
│   ├── data_understanding.py  # Data analysis agent
│   ├── query_processing.py    # NL query interpreter
│   ├── data_analysis.py       # Statistical analysis
│   ├── visualization.py       # Chart generation
│   └── insight_generation.py  # Business intelligence
├── utils/
│   ├── data_loader.py         # Data loading utilities
│   ├── cache_manager.py       # Caching system
│   ├── session_utils.py       # Session utilities
│   └── validators.py          # Input validation
├── config/
│   ├── agent_prompts.py       # System prompts for each agent
│   ├── settings.py           # Configuration settings
│   └── context_prompts.py    # Context-aware prompts
├── templates/
│   ├── chat_interface.html    # Web interface template
│   └── dashboard.html         # Dashboard template
└── sessions/                  # Session storage directory
```

## Usage Guide

### Starting a New Chat Session
```python
from agentic_data_chat import AgenticDataChat

# Initialize the system
chat_system = AgenticDataChat(api_key="your_api_key")

# Start new session (context automatically refreshed)
session_id = chat_system.start_new_session()

# Load data for the session
chat_system.load_data("path/to/your/data.csv", session_id)
```

### Conversational Data Interaction Examples

#### Session 1: Initial Data Exploration
```
User: "I've uploaded a sales dataset. What can you tell me about it?"
System: [Analyzes data structure, provides overview]

User: "Show me the sales trends over time"
System: [Creates time series visualization with insights]

User: "Which regions are performing best?"
System: [Analyzes regional performance, references previous trend analysis]

User: "Create a dashboard combining these insights"
System: [Builds dashboard using context from previous queries]
```

#### Session 2: New Session (Context Refreshed)
```
User: "I have customer data now. How does it look?"
System: [Fresh context - analyzes new dataset without prior assumptions]

User: "Compare this with sales data I analyzed before"
System: [Cannot access previous session - suggests reloading sales data]
```

#### Continued Session: Context Maintained
```
User: "Go back to the regional analysis"
System: [References previous regional analysis from current session]

User: "Now add customer demographics to that view"
System: [Combines current customer data with previous regional insights]
```

## Context Management Features

### Session Lifecycle
1. **Session Creation**: Fresh context initialized
2. **Context Building**: Accumulates conversation history
3. **Context Maintenance**: Manages memory and relevance
4. **Session Timeout**: Automatic cleanup after inactivity
5. **Context Refresh**: Clean slate for new sessions

### Context Types Maintained

#### Conversation Context
- Previous questions and answers
- Analysis results and insights
- Generated visualizations
- Follow-up suggestions

#### Data Context
- Current dataset information
- Previous transformations
- Cached analysis results
- User preferences

#### Analytical Context
- Ongoing analysis threads
- Comparative studies
- Progressive insights
- Hypothesis tracking

### Context-Aware Features

#### Reference Resolution
```
User: "Show me more details about that outlier"
System: [Knows which outlier from previous analysis]

User: "Make the same chart but for last year's data"
System: [Remembers chart type and applies to time-filtered data]
```

#### Progressive Analysis
```
User: "Analyze customer segments"
System: [Performs segmentation analysis]

User: "Now show how each segment performs in sales"
System: [Combines segmentation with sales analysis]

User: "Which segment has the highest growth potential?"
System: [Builds on previous analyses to provide growth insights]
```

## Configuration Options

### Session Management Settings
```python
SESSION_CONFIG = {
    "max_history_turns": 20,           # Maximum conversation turns to remember
    "session_timeout_hours": 2,        # Hours before session expires
    "context_compression": True,       # Compress old context for memory
    "auto_save_session": True,         # Save sessions automatically
    "context_relevance_threshold": 0.7 # Relevance score for context inclusion
}
```

### Agent Behavior Configuration
```python
AGENT_CONFIG = {
    "temperature": 0.1,                # Response creativity (0-1)
    "max_tokens": 2048,               # Maximum response length
    "context_window": 10,             # Previous turns to consider
    "enable_memory": True,            # Enable conversation memory
    "parallel_processing": True,       # Process agents concurrently
    "timeout_seconds": 30             # Query timeout
}
```

### Data Processing Settings
```python
DATA_CONFIG = {
    "max_file_size_mb": 100,          # Maximum upload size
    "max_rows": 1000000,              # Maximum rows to process
    "cache_analysis_results": True,    # Cache expensive computations
    "auto_data_profiling": True,      # Automatic data profiling
    "supported_formats": ["csv", "xlsx", "json", "parquet"]
}
```

## API Reference

### Core Classes

#### AgenticDataChat
```python
class AgenticDataChat:
    def start_new_session(self) -> str
    def load_data(self, file_path: str, session_id: str) -> bool
    def chat(self, query: str, session_id: str) -> Dict[str, Any]
    def get_session_history(self, session_id: str) -> List[Dict]
    def refresh_context(self, session_id: str) -> bool
```

#### ContextManager
```python
class ContextManager:
    def create_session(self, dataset_info: Dict = None) -> str
    def get_session_context(self, session_id: str) -> SessionContext
    def add_conversation_turn(self, session_id: str, turn: ConversationTurn)
    def refresh_session_context(self, session_id: str) -> bool
    def cleanup_expired_sessions(self) -> int
```

### Conversation Flow API
```python
# Start new conversation
response = chat_system.chat("What's in this dataset?", session_id)

# Continue conversation with context
response = chat_system.chat("Show me sales by region", session_id)

# Reference previous analysis
response = chat_system.chat("Compare that with last month", session_id)

# Get conversation history
history = chat_system.get_session_history(session_id)
```

## Advanced Features

### Multi-Turn Analysis Workflows
- **Progressive Drilling**: Start broad, get more specific
- **Comparative Analysis**: Compare across different dimensions
- **Hypothesis Testing**: Build and test hypotheses across turns
- **Insight Synthesis**: Combine insights from multiple analyses

### Smart Context Management
- **Relevance Scoring**: Prioritize most relevant context
- **Context Compression**: Summarize old context to save memory
- **Semantic Clustering**: Group related conversation threads
- **Auto-Tagging**: Tag conversations for easy retrieval

### Visualization Continuity
- **Chart Evolution**: Modify existing charts based on new requests
- **Dashboard Building**: Progressively build comprehensive dashboards
- **Style Consistency**: Maintain visual consistency across session
- **Interactive Linking**: Link visualizations based on conversation flow

## Troubleshooting

### Common Context Issues

#### Context Not Preserved
```python
# Check session status
session = chat_system.get_session_context(session_id)
if not session.is_active():
    print("Session expired, starting new session")
    session_id = chat_system.start_new_session()
```

#### Memory Issues with Long Conversations
```python
# Configure context compression
chat_system.configure_context(
    max_history_turns=10,
    enable_compression=True
)
```

#### Reference Resolution Failures
```python
# Be more explicit in queries
"Show the correlation matrix from the previous analysis"
# Instead of: "Show that correlation thing again"
```

### Debugging Context Flow
```python
# Enable debug mode for context tracking
import logging
logging.getLogger('context_manager').setLevel(logging.DEBUG)

# View context at any point
context = chat_system.get_debug_context(session_id)
print(f"Current context size: {len(context.conversation_history)}")
print(f"Available data: {context.dataset_info.keys()}")
```

## Performance Optimization

### Context Management
- Implement context compression for long conversations
- Use relevance scoring to prioritize important context
- Cache frequent analysis patterns
- Implement smart context pruning

### Memory Management
- Set appropriate history limits
- Use lazy loading for large datasets
- Implement efficient context serialization
- Monitor memory usage per session

## Security & Privacy

### Session Security
- Generate cryptographically secure session IDs
- Implement session timeout and cleanup
- Secure context storage encryption
- User authentication and authorization

### Data Privacy
- Local data processing only
- No data persistence beyond session
- Configurable data retention policies
- Audit logging for compliance

## Deployment Options

### Local Development
```bash
python main.py --mode development
```

### Production Deployment
```bash
# Using Docker
docker build -t agentic-data-chat .
docker run -p 8000:8000 agentic-data-chat

# Using cloud deployment
# Configure for your cloud platform
```

### Streamlit Interface
```bash
streamlit run app.py
```

This architecture provides a robust, context-aware conversational interface for data analysis that maintains session continuity while properly refreshing context for new sessions.