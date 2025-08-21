"""
Base agent class and common utilities for the agentic system
"""
import abc
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import google.generativeai as genai
from config.settings import Config
from utils.session_utils import SessionContext

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            
            # Configure the model
            generation_config = {
                "temperature": Config.AGENT_CONFIG["temperature"],
                "max_output_tokens": Config.AGENT_CONFIG["max_tokens"],
            }
            
            self.model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL,
                generation_config=generation_config,
                system_instruction=self.system_prompt
            )
            
            logger.info(f"Initialized {self.name} with model {Config.GEMINI_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model for {self.name}: {str(e)}")
            raise
    
    @abc.abstractmethod
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     **kwargs) -> Dict[str, Any]:
        """Process a query with the given context"""
        pass
    
    async def generate_response(self, 
                               prompt: str, 
                               context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using the Gemini model"""
        try:
            # Add context to prompt if provided
            if context:
                context_str = self._format_context(context)
                full_prompt = f"{prompt}\n\nContext:\n{context_str}"
            else:
                full_prompt = prompt
            
            # Generate response
            response = await self._async_generate(full_prompt)
            return response.text if response else "Error generating response"
            
        except Exception as e:
            logger.error(f"Error generating response in {self.name}: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _async_generate(self, prompt: str):
        """Async wrapper for model generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.model.generate_content(prompt)
        )
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a readable string"""
        context_parts = []
        
        if "conversation_history" in context:
            history = context["conversation_history"][-5:]  # Last 5 turns
            context_parts.append("Recent Conversation:")
            for turn in history:
                context_parts.append(f"User: {turn.get('user', '')}")
                context_parts.append(f"Assistant: {turn.get('assistant', '')}")
        
        if "dataset_info" in context:
            context_parts.append(f"Dataset: {context['dataset_info']}")
        
        if "previous_results" in context:
            context_parts.append(f"Previous Results: {context['previous_results']}")
        
        return "\n".join(context_parts)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "model": Config.GEMINI_MODEL,
            "status": "active" if self.model else "inactive"
        }

class AgentResponse:
    """Standardized response format for all agents"""
    
    def __init__(self, 
                 agent_name: str,
                 content: str,
                 response_type: str = "text",
                 metadata: Optional[Dict[str, Any]] = None,
                 artifacts: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        self.content = content
        self.response_type = response_type  # text, code, visualization, data
        self.metadata = metadata or {}
        self.artifacts = artifacts or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "agent_name": self.agent_name,
            "content": self.content,
            "response_type": self.response_type,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        """Create response from dictionary"""
        response = cls(
            agent_name=data["agent_name"],
            content=data["content"],
            response_type=data.get("response_type", "text"),
            metadata=data.get("metadata", {}),
            artifacts=data.get("artifacts", {})
        )
        if "timestamp" in data:
            response.timestamp = datetime.fromisoformat(data["timestamp"])
        return response

class AgentRegistry:
    """Registry for managing agent instances"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
    
    def register_agent(self, agent: BaseAgent, capabilities: List[str]):
        """Register an agent with its capabilities"""
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = capabilities
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def get_suitable_agents(self, query_type: str) -> List[BaseAgent]:
        """Get agents suitable for a query type"""
        suitable = []
        for name, capabilities in self.agent_capabilities.items():
            if query_type.lower() in [cap.lower() for cap in capabilities]:
                agent = self.agents.get(name)
                if agent:
                    suitable.append(agent)
        return suitable
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents with their info"""
        return {
            name: agent.get_agent_info() 
            for name, agent in self.agents.items()
        }

# Global agent registry
agent_registry = AgentRegistry()
