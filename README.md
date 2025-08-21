# Agentic Data Chat & Visualization System

A sophisticated multi-agent conversational data analysis and visualization system powered by Google Gemini 2.0 Flash.

## 🚀 Features

### Multi-Agent Architecture
- **Orchestrator Agent**: Master controller coordinating all agents and workflows
- **Data Understanding Agent**: Analyzes data structure, quality, and characteristics
- **Query Processing Agent**: Interprets natural language queries into data operations
- **Data Analysis Agent**: Performs statistical analysis and data science operations
- **Visualization Agent**: Creates interactive charts and visualizations
- **Insight Generation Agent**: Generates business intelligence and actionable recommendations
- **Context Manager**: Manages conversational context and session state

### Core Capabilities
- 🤖 **Conversational Data Analysis**: Natural language interaction with your data
- 📊 **Smart Visualizations**: Context-aware chart generation with Plotly
- 🔍 **Advanced Analytics**: Statistical analysis, correlations, trends, and predictions
- 💡 **Business Insights**: AI-powered insights and recommendations
- 📈 **Session Memory**: Maintains context across conversation turns
- 🔄 **Context Management**: Automatic memory optimization and reference resolution
- 📁 **Multi-format Support**: CSV, Excel, JSON, and Parquet files

### Technical Features
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **Session Management**: Persistent conversation context with automatic cleanup
- **Memory Optimization**: Intelligent context compression and management
- **Error Handling**: Comprehensive error handling and logging
- **Scalable Architecture**: Modular design for easy extension and customization

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google AI Studio API key (for Gemini 2.0 Flash)

### Quick Start

1. **Clone or Download the Project**
   ```bash
   # If using git
   git clone <repository-url>
   cd chat-to-your-database
   
   # Or download and extract the files
   ```

2. **Set Up Environment**
   - Run `start.bat` (Windows) - this will handle everything automatically
   - Or follow manual setup below

### Manual Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   LOG_LEVEL=INFO
   ```

4. **Start the Application**
   ```bash
   python main.py
   ```

The API will be available at:
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 API Usage

### Core Endpoints

#### 1. Create Session
```http
POST /session/create
Content-Type: application/json

{
    "user_preferences": {
        "detail_level": "medium",
        "preferred_interaction": "analysis"
    }
}
```

#### 2. Upload Data
```http
POST /session/{session_id}/upload
Content-Type: multipart/form-data

file: <your_data_file>
```

#### 3. Chat with Your Data
```http
POST /chat
Content-Type: application/json

{
    "session_id": "your_session_id",
    "query": "Show me the correlation between sales and marketing spend"
}
```

### Example Conversation Flow

1. **Create Session**: Get a session ID
2. **Upload Data**: Upload your CSV/Excel file
3. **Start Chatting**: Ask questions about your data

Example queries:
- "What's the structure of this dataset?"
- "Show me sales trends over time"
- "Create a correlation heatmap"
- "What are the key insights from this data?"
- "Compare performance across different regions"

## 🏗 Architecture

### Agent Hierarchy
```
Orchestrator (Master Controller)
├── Data Understanding Agent
├── Query Processing Agent  
├── Data Analysis Agent
├── Visualization Agent
├── Insight Generation Agent
└── Context Manager
```

### Session Management
- **Persistent Context**: Maintains conversation history and user preferences
- **Automatic Compression**: Optimizes memory usage for long conversations
- **Reference Resolution**: Resolves references to previous analyses
- **Smart Caching**: Caches results for improved performance

### Data Processing Pipeline
1. **Data Upload** → Data validation and profiling
2. **Query Processing** → Natural language interpretation
3. **Agent Coordination** → Route to appropriate specialized agents
4. **Result Synthesis** → Combine and format responses
5. **Context Update** → Update session state and memory

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key (required)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `SESSION_TIMEOUT_MINUTES`: Session timeout in minutes (default: 60)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)

### Model Configuration
The system uses Google Gemini 2.0 Flash by default. You can modify model settings in `config/settings.py`:
- Temperature: 0.1 (for consistent, analytical responses)
- Max tokens: 2048
- Model: gemini-2.0-flash-exp (can be changed to other Gemini models)

## 📊 Supported Data Formats

- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: JavaScript Object Notation
- **Parquet**: Columnar data format

## 🔍 System Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### System Statistics
```bash
curl http://localhost:8000/system/stats
```

### Session Management
- View session status: `GET /session/{session_id}/status`
- Compress context: `POST /session/{session_id}/context/compress`
- Get context summary: `GET /session/{session_id}/context/summary`

## 🚀 Advanced Features

### Context Compression
The system automatically compresses conversation history when memory limits are approached, maintaining essential context while optimizing performance.

### Reference Resolution
The Context Manager can resolve references like "that chart", "previous analysis", or "the correlation we found earlier".

### Intelligent Caching
Results are cached and reused when appropriate, improving response times for similar queries.

### Business Intelligence
The Insight Generation Agent provides strategic insights, recommendations, and business-focused analysis of your data.

## 🛠 Development

### Project Structure
```
chat-to-your-database/
├── agents/                    # Specialized agent implementations
├── config/                    # Configuration files and prompts
├── utils/                     # Utility functions and helpers
├── main.py                    # FastAPI application
├── orchestrator.py            # Master orchestrator agent
├── context_manager.py         # Context management system
├── requirements.txt           # Python dependencies
├── start.bat                  # Windows startup script
└── README.md                  # This file
```

### Adding New Agents
1. Create agent class inheriting from `BaseAgent`
2. Add agent to orchestrator routing
3. Update prompts in `config/agent_prompts.py`

### Extending Functionality
- Add new analysis types in `DataAnalysisAgent`
- Create new visualization types in `VisualizationAgent`
- Enhance business logic in `InsightGenerationAgent`

## 📝 Logging

Logs are written to the console by default. To enable file logging, create a `logs/` directory and modify the logging configuration in `main.py`.

## 🔐 Security Notes

- API keys should be kept secure in `.env` file
- Consider implementing authentication for production use
- CORS is currently set to allow all origins - restrict for production
- File uploads are validated for size - consider additional security measures

## 📈 Performance Tips

- Use session compression for long conversations
- Upload smaller datasets for faster processing
- Clear expired sessions regularly using `/system/cleanup`
- Monitor system stats to track resource usage

## 🤝 Contributing

This system is designed to be modular and extensible. You can:
- Add new specialized agents
- Enhance existing agent capabilities
- Improve visualization options
- Add new data source connectors

## 📄 License

[Add your license information here]

## 🆘 Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Review system logs for error details
3. Verify your API key is correctly configured
4. Ensure your data format is supported

---

**Built with**: Python, FastAPI, Google Gemini 2.0 Flash, Plotly, Pandas, and modern AI agent architecture.
