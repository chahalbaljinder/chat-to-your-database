"""
FastAPI application for Agentic Data Chat & Visualization System
"""
import asyncio
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import agents and utilities
from orchestrator import Orchestrator
from context_manager import ContextManager
from utils.session_utils import SessionManager, convert_numpy_types, session_manager
from utils.data_loader import DataLoader
from config.settings import SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Data Chat API",
    description="Multi-agent conversational data analysis and visualization system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
orchestrator = Orchestrator()
context_manager = ContextManager()
data_loader = DataLoader()

# Pydantic models for API
class ChatRequest(BaseModel):
    session_id: str
    query: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    session_id: str
    agent_responses: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    session_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DataUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_info: Optional[Dict[str, Any]] = None
    session_id: str
    error: Optional[str] = None

# Additional models for frontend endpoints
class AnalyzeRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AnalyzeResponse(BaseModel):
    message: str
    session_id: str
    query: str
    status: str = "started"

class StatusResponse(BaseModel):
    status: str  # "processing", "completed", "error"
    session_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    preview: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class QuickMenuResponse(BaseModel):
    menu_items: List[Dict[str, Any]]

# Store for background analysis tasks
analysis_tasks = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Agentic Data Chat API")
    logger.info(f"Model: {SETTINGS.MODEL_NAME}")
    logger.info(f"Session timeout: {SETTINGS.SESSION_TIMEOUT_MINUTES} minutes")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Agentic Data Chat API")
    # Cleanup any background tasks or resources

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "model": SETTINGS.MODEL_NAME
    }

# Session management endpoints
@app.post("/session/create", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new conversation session"""
    try:
        # Create new session
        session_context = session_manager.create_session(
            user_preferences=request.user_preferences or {}
        )
        
        logger.info(f"Created new session: {session_context.session_id}")
        
        return SessionResponse(
            success=True,
            session_id=session_context.session_id,
            message="Session created successfully",
            session_info={
                "created_at": session_context.created_at.isoformat(),
                "user_preferences": session_context.user_preferences
            }
        )
    
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/status", response_model=SessionResponse)
async def get_session_status(session_id: str):
    """Get session status and information"""
    try:
        context_status = context_manager.get_context_status(session_id)
        
        if "error" in context_status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            message="Session status retrieved",
            session_info=context_status
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    try:
        success = session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Deleted session: {session_id}")
        
        return {"success": True, "message": f"Session {session_id} deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Data upload endpoints
@app.post("/session/{session_id}/upload", response_model=DataUploadResponse)
async def upload_data(
    session_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a dataset for the session"""
    try:
        # Get or create session
        session_context = session_manager.get_session(session_id)
        if not session_context:
            session_context = session_manager.create_session()
            session_id = session_context.session_id
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > SETTINGS.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail=f"File size exceeds {SETTINGS.MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Save file temporarily
        temp_filename = f"temp_{session_id}_{file.filename}"
        temp_path = os.path.join("temp", temp_filename)
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load and process data
        df, dataset_info = data_loader.load_file(temp_path)
        
        # Update the dataset_info to point to the persistent temp path
        dataset_info.file_path = temp_path
        
        # Store dataset info in session context
        session_context.dataset_info = dataset_info
        
        # Update session
        session_manager.update_session(session_context)
        
        # Don't clean up temp file immediately - keep it for the session duration
        # The session manager will clean it up when the session expires
        session_context.temp_files = getattr(session_context, 'temp_files', [])
        session_context.temp_files.append(temp_path)
        
        logger.info(f"Data uploaded successfully for session {session_id}: {file.filename}")
        
        return DataUploadResponse(
            success=True,
            message=f"Dataset '{file.filename}' uploaded and processed successfully",
            dataset_info={
                **convert_numpy_types(dataset_info.to_dict()),
                "preview": {
                    "columns": dataset_info.columns,
                    "rows": df.head(5).fillna("").values.tolist() if hasattr(df, 'head') else []
                }
            },
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Frontend-specific endpoints
@app.post("/analyze", response_model=AnalyzeResponse) 
async def analyze_data(request: AnalyzeRequest):
    """Analyze data endpoint for frontend - starts background analysis"""
    try:
        # Create or get session
        if request.session_id:
            session_context = session_manager.get_session(request.session_id)
        else:
            session_context = session_manager.create_session()
        
        if not session_context:
            session_context = session_manager.create_session()
        
        session_id = session_context.session_id
        
        # Store the analysis task as "processing"
        analysis_tasks[session_id] = {
            "status": "processing",
            "query": request.query,
            "started_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Start background analysis
        async def run_analysis():
            try:
                result = await orchestrator.process_query(
                    query=request.query,
                    context=session_context
                )
                
                if result.get("success"):
                    agent_response = result.get("response")
                    analysis_tasks[session_id]["status"] = "completed"
                    analysis_tasks[session_id]["result"] = {
                        "description": "Analysis completed",
                        "textResult": agent_response.content if agent_response else "Analysis completed",
                        "chartType": "bar",  # Default chart type
                        "data": []  # Chart data would come from artifacts
                    }
                    
                    # Extract chart data if available
                    if agent_response and agent_response.artifacts:
                        artifacts = agent_response.artifacts
                        if "chart_data" in artifacts:
                            analysis_tasks[session_id]["result"]["data"] = artifacts["chart_data"]
                        if "chart_type" in artifacts:
                            analysis_tasks[session_id]["result"]["chartType"] = artifacts["chart_type"]
                else:
                    analysis_tasks[session_id]["status"] = "error"
                    analysis_tasks[session_id]["error"] = result.get("error", "Analysis failed")
                    
            except Exception as e:
                analysis_tasks[session_id]["status"] = "error"
                analysis_tasks[session_id]["error"] = str(e)
        
        # Start the background task
        asyncio.create_task(run_analysis())
        
        return AnalyzeResponse(
            message="Analysis started",
            session_id=session_id,
            query=request.query,
            status="started"
        )
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """Get status of analysis task"""
    try:
        if session_id not in analysis_tasks:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        task = analysis_tasks[session_id]
        
        response = StatusResponse(
            status=task["status"],
            session_id=session_id,
            result=task.get("result"),
            error=task.get("error")
        )
        
        # Clean up completed or errored tasks after 1 minute
        if task["status"] in ["completed", "error"]:
            started_time = task["started_at"]
            if (datetime.now() - started_time).seconds > 60:
                del analysis_tasks[session_id]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file_simple(file: UploadFile = File(...)):
    """Simple file upload endpoint for frontend"""
    try:
        # Create a new session for this upload
        session_context = session_manager.create_session()
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        content = await file.read()
        file_size = len(content)
        
        if file_size > SETTINGS.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail=f"File size exceeds {SETTINGS.MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Save file temporarily
        temp_filename = f"temp_{session_context.session_id}_{file.filename}"
        temp_path = os.path.join("temp", temp_filename)
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        try:
            # Load and process data
            df, dataset_info = data_loader.load_file(temp_path)
            
            # Store dataset info in session context
            dataset_info.file_path = temp_path
            session_context.dataset_info = dataset_info
            
            # Keep temp file for session
            session_context.temp_files = [temp_path]
            
            # Update session
            session_manager.update_session(session_context)
            
            # Create preview data
            preview_data = {
                "columns": dataset_info.columns,
                "rows": df.head(5).fillna("").values.tolist() if hasattr(df, 'head') else []
            }
            
            logger.info(f"File uploaded successfully: {file.filename}")
            
            return UploadResponse(
                success=True,
                message=f"File '{file.filename}' uploaded successfully",
                preview=preview_data
            )
            
        except Exception as processing_error:
            # If processing fails, still return success but with limited preview
            logger.warning(f"File processing failed: {str(processing_error)}")
            
            # Try to create a simple preview
            try:
                content_str = content.decode('utf-8')
                lines = content_str.split('\n')[:5]
                preview_data = {
                    "columns": ["Content"],
                    "rows": [[line[:50] + "..." if len(line) > 50 else line] for line in lines if line.strip()]
                }
            except:
                preview_data = {
                    "columns": ["File"],
                    "rows": [[f"Uploaded: {file.filename}"]]
                }
            
            return UploadResponse(
                success=True,
                message=f"File '{file.filename}' uploaded (processing had issues)",
                preview=preview_data
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return UploadResponse(
            success=False,
            message="Upload failed",
            error=str(e)
        )

@app.get("/quick-menu", response_model=QuickMenuResponse)
async def get_quick_menu():
    """Get quick analysis menu items"""
    try:
        menu_items = [
            {
                "id": "overview",
                "title": "📊 Data Overview",
                "description": "Get a comprehensive summary of your dataset structure and content",
                "query": "Show me a complete overview of this dataset including structure, statistics, and key insights"
            },
            {
                "id": "statistics", 
                "title": "📈 Statistical Summary",
                "description": "View descriptive statistics for all numeric columns",
                "query": "Calculate and show descriptive statistics for all numeric columns"
            },
            {
                "id": "correlations",
                "title": "🔗 Correlation Analysis", 
                "description": "Find relationships and correlations between variables",
                "query": "Show correlation analysis between all numeric variables with a heatmap"
            },
            {
                "id": "trends",
                "title": "📉 Trend Analysis",
                "description": "Identify patterns and trends over time",
                "query": "Analyze trends and patterns in the data over time"
            },
            {
                "id": "missing_data",
                "title": "❓ Missing Data Analysis",
                "description": "Analyze missing values and data quality issues",
                "query": "Analyze missing values and data quality issues in the dataset"
            },
            {
                "id": "distributions",
                "title": "📊 Distribution Analysis",
                "description": "Visualize distributions of key variables",
                "query": "Show distribution plots for the most important variables"
            },
            {
                "id": "outliers",
                "title": "🎯 Outlier Detection",
                "description": "Identify and analyze outliers in the data",
                "query": "Detect and analyze outliers in the numeric columns"
            },
            {
                "id": "insights",
                "title": "💡 Key Insights",
                "description": "Generate business insights and recommendations",
                "query": "What are the key insights and actionable recommendations from this data?"
            }
        ]
        
        return QuickMenuResponse(menu_items=menu_items)
        
    except Exception as e:
        logger.error(f"Error getting quick menu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for conversational data analysis"""
    try:
        # Get session context
        session_context = session_manager.get_session(request.session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update context if provided
        if request.context:
            session_context.metadata.update(request.context)
        
        # Process query through orchestrator
        result = await orchestrator.process_query(
            query=request.query,
            context=session_context
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Processing failed")
            )
        
        # Extract response components
        agent_response = result.get("response")
        agent_responses = result.get("agent_responses", [])
        
        # Prepare response with numpy type conversion
        response_data = {
            "success": True,
            "response": agent_response.content if agent_response else "No response generated",
            "session_id": request.session_id,
            "agent_responses": agent_responses,
            "metadata": agent_response.metadata if agent_response else None,
            "artifacts": agent_response.artifacts if agent_response else None
        }
        
        # Convert all numpy types for JSON serialization
        response_data = convert_numpy_types(response_data)
        
        response = ChatResponse(**response_data)
        
        logger.info(f"Chat query processed for session {request.session_id}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return ChatResponse(
            success=False,
            response="",
            session_id=request.session_id,
            agent_responses=[],
            error=str(e)
        )

# Context management endpoints
@app.post("/session/{session_id}/context/compress")
async def compress_session_context(session_id: str):
    """Compress session context to save memory"""
    try:
        session_context = session_manager.get_session(session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        result = await context_manager.process(
            query="compress",
            context=session_context,
            operation="compress"
        )
        
        return {
            "success": result.get("success", False),
            "message": "Context compression completed",
            "details": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error compressing context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/context/summary")
async def get_context_summary(session_id: str):
    """Get a summary of the session context"""
    try:
        session_context = session_manager.get_session(session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        summary = {
            "session_id": session_id,
            "conversation_turns": len(session_context.conversation_history),
            "cached_results": len(session_context.cached_results),
            "user_preferences": session_context.user_preferences,
            "dataset_loaded": session_context.dataset_info is not None,
            "created_at": session_context.created_at.isoformat(),
            "last_active": session_context.last_active.isoformat()
        }
        
        return {"success": True, "summary": summary}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting context summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# System management endpoints
@app.post("/system/cleanup")
async def cleanup_system():
    """Clean up expired sessions and temporary files"""
    try:
        # Clean up expired sessions
        cleanup_result = context_manager.cleanup_expired_sessions()
        
        # Clean up temp directory
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            temp_files = os.listdir(temp_dir)
            for temp_file in temp_files:
                temp_path = os.path.join(temp_dir, temp_file)
                # Remove files older than 1 hour
                if os.path.getctime(temp_path) < (datetime.now().timestamp() - 3600):
                    os.remove(temp_path)
        
        return {
            "success": True,
            "message": "System cleanup completed",
            "cleaned_sessions": cleanup_result.get("cleaned_sessions", 0)
        }
    
    except Exception as e:
        logger.error(f"Error during system cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        active_sessions = len(session_manager.active_sessions)
        
        return {
            "active_sessions": active_sessions,
            "model": SETTINGS.MODEL_NAME,
            "session_timeout_minutes": SETTINGS.SESSION_TIMEOUT_MINUTES,
            "max_file_size_mb": SETTINGS.MAX_FILE_SIZE_MB,
            "uptime": "Runtime stats not implemented"  # Could add actual uptime tracking
        }
    
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility functions
async def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {file_path}: {str(e)}")

# WebSocket endpoint for real-time chat (optional)
# This could be added for real-time streaming responses
# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await websocket.accept()
#     # Implement WebSocket chat functionality

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
