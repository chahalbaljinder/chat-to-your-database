
# Frontend-Backend API Endpoint Mapping

This document maps all frontend requests to backend API endpoints in the Agentic Data Chat system.

## 📋 Endpoint Summary

### ✅ Mapped Endpoints

| Frontend Usage | Backend Endpoint | Method | Purpose |
|---------------|------------------|---------|---------|
| Main chat analysis | `POST /analyze` | POST | Start background data analysis |
| Analysis status polling | `GET /status/{session_id}` | GET | Check analysis completion status |
| File upload | `POST /upload` | POST | Simple file upload with preview |
| Quick analysis menu | `GET /quick-menu` | GET | Get predefined analysis queries |
| Health check | `GET /health` | GET | System health status |

### 🔄 Existing Backend Endpoints (Also Available)

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `POST /session/create` | POST | Create new session with preferences |
| `GET /session/{session_id}/status` | GET | Get detailed session information |
| `DELETE /session/{session_id}` | DELETE | Delete a session |
| `POST /session/{session_id}/upload` | POST | Upload file to specific session |
| `POST /chat` | POST | Direct chat interface (alternative to /analyze) |
| `POST /session/{session_id}/context/compress` | POST | Compress session context |
| `GET /session/{session_id}/context/summary` | GET | Get context summary |
| `POST /system/cleanup` | POST | System cleanup |
| `GET /system/stats` | GET | System statistics |

## 🔗 Frontend Component Mapping

### App.js
- **File Upload**: Uses `POST /upload` 
- **Chat Analysis**: Uses `POST /analyze` → `GET /status/{session_id}`
- **Health Status**: Implicitly uses backend health

### FileUpload.js
- **Upload Handler**: `POST /upload`
- **Supported Formats**: CSV, Excel, Text, Word files

### ChatMessage.js  
- **Message Display**: Renders responses from `/analyze` and `/status` endpoints
- **Chart Integration**: Displays chart data from analysis results

### ChartDisplay.js
- **Visualization**: Renders chart data from analysis artifacts
- **Chart Types**: Bar, Line, Pie, Area charts using Recharts

### DataPreview.js
- **Preview Data**: Shows data structure from upload response
- **Column Display**: Lists columns and sample rows

### QuickMenu.js
- **Menu Items**: Fetches from `GET /quick-menu`
- **Query Selection**: Sends queries to `/analyze`

## 📊 Data Flow

### Upload Flow
1. **Frontend**: User drops/selects file
2. **POST /upload**: File uploaded to backend
3. **Response**: Preview data returned
4. **Frontend**: Display data preview in sidebar

### Analysis Flow  
1. **Frontend**: User types query or selects from quick menu
2. **POST /analyze**: Query sent to backend, returns session_id
3. **Background**: Backend processes query asynchronously  
4. **GET /status/{session_id}**: Frontend polls every 2 seconds
5. **Response**: When complete, returns results with chart data
6. **Frontend**: Display results and charts

### Quick Menu Flow
1. **Frontend**: Component loads
2. **GET /quick-menu**: Fetch predefined queries
3. **Response**: List of analysis templates
4. **Frontend**: Display as clickable menu items

## 🛠 Response Formats

### `/analyze` Response
```json
{
  "message": "Analysis started",
  "session_id": "uuid-string", 
  "query": "user query",
  "status": "started"
}
```

### `/status/{session_id}` Response  
```json
{
  "status": "completed",
  "session_id": "uuid-string",
  "result": {
    "description": "Analysis completed",
    "textResult": "Analysis text response",
    "chartType": "bar",
    "data": [{"name": "A", "value": 100}]
  }
}
```

### `/upload` Response
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "preview": {
    "columns": ["col1", "col2"],
    "rows": [["val1", "val2"]]
  }
}
```

### `/quick-menu` Response
```json
{
  "menu_items": [
    {
      "id": "overview",
      "title": "📊 Data Overview", 
      "description": "Get dataset summary",
      "query": "Show me dataset overview"
    }
  ]
}
```

## ⚡ Key Features

### Asynchronous Processing
- Analysis runs in background using Python `asyncio.create_task()`
- Frontend polls status every 2 seconds until completion
- Tasks auto-cleanup after 1 minute when complete

### Session Management  
- Each analysis gets a unique session ID
- Sessions store dataset info and analysis context
- Automatic session cleanup for expired sessions

### Error Handling
- All endpoints have try/catch with proper HTTP status codes
- Frontend shows user-friendly error messages
- Backend logs detailed error information

### Chart Integration
- Analysis results include chart type and data
- Frontend renders using Recharts library
- Supports bar, line, pie, and area charts

## 🚀 Usage Examples

### Upload and Analyze Workflow
```javascript
// 1. Upload file
const uploadResponse = await fetch('/upload', {
  method: 'POST', 
  body: formData
});

// 2. Start analysis  
const analyzeResponse = await fetch('/analyze', {
  method: 'POST',
  body: JSON.stringify({query: "Show data overview"})
});

// 3. Poll for results
const statusResponse = await fetch(`/status/${sessionId}`);
```

### Quick Menu Integration
```javascript
// Fetch menu items
const menuResponse = await fetch('/quick-menu');
const {menu_items} = await menuResponse.json();

// Use menu query
const selectedQuery = menu_items[0].query;
// Send to /analyze endpoint
```

This mapping ensures the React frontend can fully utilize all backend capabilities while maintaining a clean, intuitive user interface.
