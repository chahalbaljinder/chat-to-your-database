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
            dataset_info=convert_numpy_types(dataset_info.to_dict()),
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
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
