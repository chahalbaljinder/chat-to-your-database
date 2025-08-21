"""
Session management utilities and context handling
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import numpy as np
from config.settings import Config

logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization"""
    import pandas as pd
    
    # Handle None and basic types first
    if obj is None:
        return None
    
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.complexfloating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.datetime64):
        return str(obj)
    elif isinstance(obj, np.timedelta64):
        return str(obj)
    elif isinstance(obj, np.bytes_):
        return obj.decode('utf-8')
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict('records'))
    elif hasattr(obj, 'dtype') and hasattr(obj.dtype, 'name'):
        # Handle pandas/numpy objects with dtype
        if pd.isna(obj):
            return None
        try:
            return convert_numpy_types(obj.item()) if hasattr(obj, 'item') else str(obj)
        except (ValueError, TypeError):
            return str(obj)
    
    # Handle collections
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(item) for item in obj]
    
    # Handle numpy dtypes themselves
    elif str(type(obj)).startswith("<class 'numpy.dtypes"):
        return str(obj)
    elif hasattr(obj, '__array__'):
        return convert_numpy_types(np.asarray(obj))
    
    # For any other object, try to convert to string as fallback
    try:
        # Check if it's JSON serializable first
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    user_query: str
    assistant_response: str
    timestamp: datetime
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp.isoformat(),
            "agent_responses": self.agent_responses,
            "metadata": self.metadata
        }
        
        # Convert numpy types to native Python types
        data = convert_numpy_types(data)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        return cls(
            user_query=data["user_query"],
            assistant_response=data["assistant_response"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_responses=data.get("agent_responses", []),
            metadata=data.get("metadata", {})
        )

@dataclass
class DatasetInfo:
    """Information about the current dataset"""
    name: str
    file_path: Optional[str] = None
    shape: tuple = (0, 0)
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    quality_info: Dict[str, Any] = field(default_factory=dict)
    last_analyzed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.last_analyzed:
            data["last_analyzed"] = self.last_analyzed.isoformat()
        
        # Convert numpy types to native Python types
        data = convert_numpy_types(data)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """Create from dictionary"""
        if "last_analyzed" in data and data["last_analyzed"]:
            data["last_analyzed"] = datetime.fromisoformat(data["last_analyzed"])
        return cls(**data)

@dataclass
class SessionContext:
    """Complete session context including conversation and data state"""
    session_id: str
    created_at: datetime
    last_active: datetime
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    dataset_info: Optional[DatasetInfo] = None
    cached_results: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    temp_files: List[str] = field(default_factory=list)  # Track temporary files for cleanup
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired"""
        timeout = timedelta(hours=Config.SESSION_TIMEOUT_HOURS)
        return datetime.now() - self.last_active > timeout
    
    @property
    def conversation_summary(self) -> str:
        """Get a summary of recent conversation"""
        if not self.conversation_history:
            return "No conversation history"
        
        recent_turns = self.conversation_history[-3:]  # Last 3 turns
        summary_parts = []
        
        for turn in recent_turns:
            summary_parts.append(f"Q: {turn.user_query[:100]}...")
            summary_parts.append(f"A: {turn.assistant_response[:100]}...")
        
        return "\n".join(summary_parts)
    
    def add_conversation_turn(self, turn: ConversationTurn):
        """Add a conversation turn and manage history size"""
        self.conversation_history.append(turn)
        self.last_active = datetime.now()
        
        # Limit history size
        if len(self.conversation_history) > Config.MAX_HISTORY_TURNS:
            # Remove oldest turns but keep them compressed
            removed_turns = self.conversation_history[:-Config.MAX_HISTORY_TURNS]
            self.conversation_history = self.conversation_history[-Config.MAX_HISTORY_TURNS:]
            
            # Store compressed summary
            if "compressed_history" not in self.metadata:
                self.metadata["compressed_history"] = []
            
            self.metadata["compressed_history"].extend([
                f"Q: {turn.user_query[:50]}... A: {turn.assistant_response[:50]}..."
                for turn in removed_turns
            ])
    
    def get_recent_context(self, num_turns: int = 5) -> Dict[str, Any]:
        """Get recent context for agent consumption"""
        recent_turns = self.conversation_history[-num_turns:] if self.conversation_history else []
        
        context = {
            "session_id": self.session_id,
            "conversation_history": [turn.to_dict() for turn in recent_turns],
            "dataset_info": self.dataset_info.to_dict() if self.dataset_info else None,
            "user_preferences": self.user_preferences,
            "cached_results": list(self.cached_results.keys())  # Just keys for brevity
        }
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "dataset_info": self.dataset_info.to_dict() if self.dataset_info else None,
            "cached_results": self.cached_results,
            "user_preferences": self.user_preferences,
            "metadata": self.metadata,
            "temp_files": self.temp_files
        }
        
        # Convert numpy types to native Python types
        data = convert_numpy_types(data)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        """Create from dictionary"""
        context = cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            cached_results=data.get("cached_results", {}),
            user_preferences=data.get("user_preferences", {}),
            metadata=data.get("metadata", {}),
            temp_files=data.get("temp_files", [])
        )
        
        # Restore conversation history
        if "conversation_history" in data:
            context.conversation_history = [
                ConversationTurn.from_dict(turn_data) 
                for turn_data in data["conversation_history"]
            ]
        
        # Restore dataset info
        if data.get("dataset_info"):
            context.dataset_info = DatasetInfo.from_dict(data["dataset_info"])
        
        return context

class SessionManager:
    """Manages session storage and retrieval"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or Config.SESSION_STORAGE_PATH)
        self.storage_path.mkdir(exist_ok=True)
        self._active_sessions: Dict[str, SessionContext] = {}
    
    def create_session(self, 
                      dataset_info: Optional[DatasetInfo] = None,
                      user_preferences: Optional[Dict[str, Any]] = None) -> SessionContext:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        context = SessionContext(
            session_id=session_id,
            created_at=now,
            last_active=now,
            dataset_info=dataset_info,
            user_preferences=user_preferences or {}
        )
        
        self._active_sessions[session_id] = context
        self._save_session(context)
        
        logger.info(f"Created new session: {session_id}")
        return context
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session by ID"""
        # Try active sessions first
        if session_id in self._active_sessions:
            context = self._active_sessions[session_id]
            if not context.is_expired:
                return context
            else:
                # Session expired, remove from active
                del self._active_sessions[session_id]
                return None
        
        # Try to load from disk
        return self._load_session(session_id)
    
    def update_session(self, context: SessionContext):
        """Update session context"""
        context.last_active = datetime.now()
        self._active_sessions[context.session_id] = context
        self._save_session(context)
    
    def _save_session(self, context: SessionContext):
        """Save session to disk"""
        try:
            session_file = self.storage_path / f"{context.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {context.session_id}: {str(e)}")
    
    def _load_session(self, session_id: str) -> Optional[SessionContext]:
        """Load session from disk"""
        try:
            session_file = self.storage_path / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            context = SessionContext.from_dict(data)
            
            # Check if expired
            if context.is_expired:
                return None
            
            # Add to active sessions
            self._active_sessions[session_id] = context
            return context
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and clean up their temporary files"""
        expired_count = 0
        
        # Clean up active sessions
        expired_active = [
            sid for sid, context in self._active_sessions.items()
            if context.is_expired
        ]
        
        for sid in expired_active:
            context = self._active_sessions[sid]
            # Clean up temporary files
            self._cleanup_temp_files(context)
            del self._active_sessions[sid]
            expired_count += 1
        
        # Clean up disk sessions (optional - could keep for analytics)
        # This is commented out to preserve session history
        """
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                last_active = datetime.fromisoformat(data["last_active"])
                timeout = timedelta(hours=Config.SESSION_TIMEOUT_HOURS * 24)  # Keep longer on disk
                
                if datetime.now() - last_active > timeout:
                    session_file.unlink()
                    expired_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking session file {session_file}: {str(e)}")
        """
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    def _cleanup_temp_files(self, context: SessionContext):
        """Clean up temporary files associated with a session"""
        import os
        
        for temp_file in context.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {str(e)}")
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs"""
        self.cleanup_expired_sessions()
        return list(self._active_sessions.keys())

# Global session manager instance
session_manager = SessionManager()
