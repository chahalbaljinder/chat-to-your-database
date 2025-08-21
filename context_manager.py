"""
Context Manager - Manages conversational context and session state
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext, ConversationTurn, session_manager
from config.agent_prompts import CONTEXT_MANAGER_PROMPT

logger = logging.getLogger(__name__)

class ContextManager(BaseAgent):
    """Agent responsible for managing conversational context and session state"""
    
    def __init__(self):
        super().__init__(
            name="ContextManager",
            system_prompt=CONTEXT_MANAGER_PROMPT
        )
        self.session_manager = session_manager
    
    async def process(self, 
                     query: str, 
                     context: SessionContext,
                     operation: str = "manage",
                     **kwargs) -> Dict[str, Any]:
        """Process context management operations"""
        
        try:
            if operation == "manage":
                return await self._manage_context(query, context, **kwargs)
            elif operation == "compress":
                return await self._compress_context(context, **kwargs)
            elif operation == "resolve_references":
                return await self._resolve_references(query, context, **kwargs)
            elif operation == "update_preferences":
                return await self._update_user_preferences(context, **kwargs)
            elif operation == "cleanup":
                return await self._cleanup_context(context, **kwargs)
            else:
                return await self._general_context_management(query, context, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in ContextManager: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error managing context: {str(e)}",
                    response_type="error"
                )
            }
    
    async def _manage_context(self, 
                            query: str, 
                            context: SessionContext, 
                            user_query: Optional[str] = None,
                            assistant_response: Optional[str] = None,
                            agent_responses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Manage context for a conversation turn"""
        
        # Add conversation turn if response provided
        if user_query and assistant_response:
            turn = ConversationTurn(
                user_query=user_query,
                assistant_response=assistant_response,
                timestamp=datetime.now(),
                agent_responses=agent_responses or [],
                metadata={"context_managed": True}
            )
            context.add_conversation_turn(turn)
        
        # Assess context quality and needs
        context_assessment = await self._assess_context_quality(context)
        
        # Determine if compression is needed
        needs_compression = self._needs_context_compression(context)
        
        # Update context relevance scores
        self._update_relevance_scores(context)
        
        # Generate context management insights
        management_prompt = f"""
        Manage context for this conversation turn:
        
        Current Query: {query}
        Session Length: {len(context.conversation_history)} turns
        
        Context Assessment:
        {context_assessment}
        
        Context Management Needs:
        - Compression needed: {needs_compression}
        - Memory usage: {self._estimate_context_memory(context)} units
        - Relevance distribution: {self._analyze_relevance_distribution(context)}
        
        Please provide:
        1. Context management recommendations
        2. Information to prioritize or archive
        3. User preference updates if applicable
        4. Session optimization suggestions
        5. Memory management recommendations
        
        Focus on maintaining conversational continuity while optimizing performance.
        """
        
        response_text = await self.generate_response(
            management_prompt,
            context.get_recent_context()
        )
        
        # Perform automatic context optimization if needed
        optimization_actions = []
        if needs_compression:
            compression_result = await self._compress_context(context)
            optimization_actions.append("Context compressed")
        
        # Update session
        self.session_manager.update_session(context)
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="context_management",
                metadata={
                    "context_size": len(context.conversation_history),
                    "needs_compression": needs_compression,
                    "optimization_actions": optimization_actions
                },
                artifacts={
                    "context_assessment": context_assessment,
                    "management_actions": optimization_actions
                }
            )
        }
    
    async def _compress_context(self, 
                              context: SessionContext, 
                              compression_ratio: float = 0.5) -> Dict[str, Any]:
        """Compress conversation context to save memory"""
        
        if len(context.conversation_history) <= 5:
            return {
                "success": True,
                "message": "No compression needed - context is already small"
            }
        
        # Identify turns to compress
        turns_to_keep = max(5, int(len(context.conversation_history) * compression_ratio))
        turns_to_compress = context.conversation_history[:-turns_to_keep]
        
        if not turns_to_compress:
            return {
                "success": True,
                "message": "No turns available for compression"
            }
        
        # Generate compression summary
        compression_prompt = f"""
        Create a concise summary of these conversation turns for context compression:
        
        Turns to compress ({len(turns_to_compress)} turns):
        {self._format_turns_for_compression(turns_to_compress)}
        
        Please provide:
        1. A concise summary of the key topics discussed
        2. Important findings or insights from these turns
        3. Decisions or conclusions reached
        4. Context that might be referenced later
        
        Keep the summary informative but concise to save memory while preserving essential context.
        """
        
        compression_summary = await self.generate_response(compression_prompt, {})
        
        # Store compressed summary
        if "compressed_history" not in context.metadata:
            context.metadata["compressed_history"] = []
        
        context.metadata["compressed_history"].append({
            "summary": compression_summary,
            "original_turns": len(turns_to_compress),
            "compressed_at": datetime.now().isoformat(),
            "topics": self._extract_topics_from_turns(turns_to_compress)
        })
        
        # Remove compressed turns
        context.conversation_history = context.conversation_history[-turns_to_keep:]
        
        logger.info(f"Compressed {len(turns_to_compress)} turns into summary for session {context.session_id}")
        
        return {
            "success": True,
            "compressed_turns": len(turns_to_compress),
            "remaining_turns": len(context.conversation_history),
            "compression_summary": compression_summary
        }
    
    async def _resolve_references(self, 
                                query: str, 
                                context: SessionContext,
                                reference_terms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Resolve references in user queries to previous context"""
        
        # Auto-detect reference terms if not provided
        if reference_terms is None:
            reference_terms = self._detect_reference_terms(query)
        
        if not reference_terms:
            return {
                "success": True,
                "message": "No references detected in query"
            }
        
        # Analyze context for reference resolution
        recent_context = context.get_recent_context(num_turns=10)
        compressed_context = context.metadata.get("compressed_history", [])
        
        resolution_prompt = f"""
        Resolve these references in the user query:
        
        User Query: {query}
        Reference Terms Detected: {', '.join(reference_terms)}
        
        Recent Context (last 10 turns):
        {self._format_context_for_references(recent_context)}
        
        Compressed History Available:
        {len(compressed_context)} compressed summaries
        
        Please identify:
        1. What each reference term likely refers to
        2. The specific context or result being referenced
        3. When the referenced item was discussed
        4. Any ambiguities that need clarification
        5. Resolved version of the query with clear references
        
        Provide clear reference resolution to maintain conversation continuity.
        """
        
        response_text = await self.generate_response(resolution_prompt, recent_context)
        
        # Extract resolved references
        resolved_references = self._extract_resolved_references(response_text, reference_terms)
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="reference_resolution",
                metadata={
                    "reference_terms": reference_terms,
                    "resolution_confidence": self._calculate_resolution_confidence(resolved_references)
                },
                artifacts={"resolved_references": resolved_references}
            )
        }
    
    async def _update_user_preferences(self, 
                                     context: SessionContext,
                                     preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update user preferences based on conversation patterns"""
        
        if preferences:
            context.user_preferences.update(preferences)
        else:
            # Infer preferences from conversation history
            inferred_prefs = self._infer_user_preferences(context)
            context.user_preferences.update(inferred_prefs)
        
        # Update session
        self.session_manager.update_session(context)
        
        return {
            "success": True,
            "updated_preferences": context.user_preferences,
            "message": f"Updated {len(context.user_preferences)} user preferences"
        }
    
    async def _cleanup_context(self, 
                             context: SessionContext,
                             cleanup_type: str = "expired") -> Dict[str, Any]:
        """Clean up context based on specified criteria"""
        
        cleanup_actions = []
        
        if cleanup_type == "expired":
            # Remove expired cached results
            current_time = datetime.now()
            expired_keys = []
            
            for key, cached_item in context.cached_results.items():
                if isinstance(cached_item, dict) and "timestamp" in cached_item:
                    cache_time = datetime.fromisoformat(cached_item["timestamp"])
                    if current_time - cache_time > timedelta(hours=1):  # 1 hour expiry
                        expired_keys.append(key)
            
            for key in expired_keys:
                del context.cached_results[key]
                cleanup_actions.append(f"Removed expired cache: {key}")
        
        elif cleanup_type == "memory":
            # Aggressive memory cleanup
            if len(context.conversation_history) > 10:
                compression_result = await self._compress_context(context, compression_ratio=0.3)
                cleanup_actions.append(f"Compressed context: {compression_result.get('compressed_turns', 0)} turns")
            
            # Clear old cached results
            if len(context.cached_results) > 20:
                # Keep only 10 most recent
                sorted_keys = sorted(
                    context.cached_results.keys(),
                    key=lambda k: context.cached_results[k].get("timestamp", ""),
                    reverse=True
                )
                keys_to_keep = sorted_keys[:10]
                keys_to_remove = [k for k in context.cached_results.keys() if k not in keys_to_keep]
                
                for key in keys_to_remove:
                    del context.cached_results[key]
                
                cleanup_actions.append(f"Cleaned {len(keys_to_remove)} cached results")
        
        # Update session
        self.session_manager.update_session(context)
        
        return {
            "success": True,
            "cleanup_actions": cleanup_actions,
            "message": f"Performed {len(cleanup_actions)} cleanup actions"
        }
    
    async def _general_context_management(self, 
                                        query: str, 
                                        context: SessionContext) -> Dict[str, Any]:
        """Handle general context management requests"""
        
        context_status = {
            "session_age": (datetime.now() - context.created_at).total_seconds() / 3600,  # hours
            "conversation_turns": len(context.conversation_history),
            "cached_items": len(context.cached_results),
            "user_preferences": len(context.user_preferences),
            "compressed_summaries": len(context.metadata.get("compressed_history", []))
        }
        
        management_prompt = f"""
        Provide context management guidance:
        
        User Query: {query}
        
        Context Status:
        {context_status}
        
        Session Health:
        - Active for: {context_status['session_age']:.1f} hours
        - Conversation depth: {context_status['conversation_turns']} turns
        - Memory usage: {self._estimate_context_memory(context)} units
        
        Please provide:
        1. Assessment of context health
        2. Recommendations for optimization
        3. Session management suggestions
        4. User experience improvements
        
        Focus on maintaining optimal conversational experience.
        """
        
        response_text = await self.generate_response(management_prompt, context.get_recent_context())
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="context_guidance",
                metadata=context_status
            )
        }
    
    # Helper methods for context management
    async def _assess_context_quality(self, context: SessionContext) -> Dict[str, Any]:
        """Assess the quality and health of the current context"""
        
        assessment = {
            "size": len(context.conversation_history),
            "age_hours": (datetime.now() - context.created_at).total_seconds() / 3600,
            "coherence": self._assess_context_coherence(context),
            "completeness": self._assess_context_completeness(context),
            "relevance": self._assess_average_relevance(context)
        }
        
        # Overall health score
        health_score = (
            min(assessment["coherence"], 1.0) * 0.3 +
            min(assessment["completeness"], 1.0) * 0.3 +
            min(assessment["relevance"], 1.0) * 0.4
        )
        
        assessment["health_score"] = health_score
        assessment["status"] = "healthy" if health_score > 0.7 else "needs_attention" if health_score > 0.4 else "poor"
        
        return assessment
    
    def _needs_context_compression(self, context: SessionContext) -> bool:
        """Determine if context compression is needed"""
        return (
            len(context.conversation_history) > 15 or
            self._estimate_context_memory(context) > 1000 or
            (datetime.now() - context.created_at).total_seconds() > 7200  # 2 hours
        )
    
    def _update_relevance_scores(self, context: SessionContext):
        """Update relevance scores for conversation turns"""
        current_time = datetime.now()
        
        for i, turn in enumerate(context.conversation_history):
            # Base relevance on recency (more recent = higher relevance)
            recency_score = 1.0 - (i / len(context.conversation_history)) * 0.5
            
            # Adjust for time decay
            age_hours = (current_time - turn.timestamp).total_seconds() / 3600
            time_decay = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            
            # Store in metadata
            if "relevance_score" not in turn.metadata:
                turn.metadata["relevance_score"] = recency_score * time_decay
    
    def _estimate_context_memory(self, context: SessionContext) -> int:
        """Estimate memory usage of context (simplified metric)"""
        memory_units = 0
        
        # Conversation history
        for turn in context.conversation_history:
            memory_units += len(turn.user_query) + len(turn.assistant_response)
            memory_units += sum(len(str(resp)) for resp in turn.agent_responses)
        
        # Cached results
        for cached_item in context.cached_results.values():
            memory_units += len(str(cached_item))
        
        # Metadata
        memory_units += len(str(context.metadata))
        
        return memory_units // 100  # Simplified units
    
    def _analyze_relevance_distribution(self, context: SessionContext) -> Dict[str, float]:
        """Analyze distribution of relevance scores"""
        if not context.conversation_history:
            return {"average": 0, "high": 0, "medium": 0, "low": 0}
        
        relevance_scores = [
            turn.metadata.get("relevance_score", 0.5)
            for turn in context.conversation_history
        ]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        high_relevance = len([s for s in relevance_scores if s > 0.8])
        medium_relevance = len([s for s in relevance_scores if 0.5 <= s <= 0.8])
        low_relevance = len([s for s in relevance_scores if s < 0.5])
        
        total = len(relevance_scores)
        
        return {
            "average": avg_relevance,
            "high": high_relevance / total if total > 0 else 0,
            "medium": medium_relevance / total if total > 0 else 0,
            "low": low_relevance / total if total > 0 else 0
        }
    
    def _format_turns_for_compression(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns for compression summary"""
        formatted_turns = []
        
        for i, turn in enumerate(turns):
            formatted_turns.append(f"Turn {i+1}:")
            formatted_turns.append(f"  User: {turn.user_query[:100]}...")
            formatted_turns.append(f"  Assistant: {turn.assistant_response[:100]}...")
            formatted_turns.append("")
        
        return '\n'.join(formatted_turns)
    
    def _extract_topics_from_turns(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract main topics from conversation turns"""
        topics = set()
        
        for turn in turns:
            # Simple topic extraction based on keywords
            query_lower = turn.user_query.lower()
            
            if any(word in query_lower for word in ["correlation", "corr", "relationship"]):
                topics.add("correlation_analysis")
            if any(word in query_lower for word in ["plot", "chart", "visualize", "graph"]):
                topics.add("visualization")
            if any(word in query_lower for word in ["trend", "time", "over time"]):
                topics.add("trend_analysis")
            if any(word in query_lower for word in ["describe", "summary", "overview"]):
                topics.add("data_exploration")
            if any(word in query_lower for word in ["compare", "vs", "versus"]):
                topics.add("comparison")
        
        return list(topics)
    
    def _detect_reference_terms(self, query: str) -> List[str]:
        """Detect reference terms in user query"""
        reference_patterns = [
            "that", "it", "this", "these", "those",
            "previous", "last", "earlier", "above", "before",
            "same", "similar", "like that", "from before",
            "the analysis", "the chart", "the result"
        ]
        
        query_lower = query.lower()
        found_references = []
        
        for pattern in reference_patterns:
            if pattern in query_lower:
                found_references.append(pattern)
        
        return found_references
    
    def _format_context_for_references(self, context: Dict[str, Any]) -> str:
        """Format context for reference resolution"""
        formatted_parts = []
        
        if "conversation_history" in context:
            for i, turn in enumerate(context["conversation_history"][-5:], 1):
                formatted_parts.append(f"Recent Turn {i}:")
                formatted_parts.append(f"  User: {turn['user_query'][:100]}...")
                formatted_parts.append(f"  Assistant: {turn['assistant_response'][:100]}...")
        
        return '\n'.join(formatted_parts)
    
    def _extract_resolved_references(self, response_text: str, reference_terms: List[str]) -> Dict[str, str]:
        """Extract resolved references from response"""
        resolved = {}
        
        # Simple pattern matching to extract resolutions
        for term in reference_terms:
            if f'"{term}"' in response_text:
                # Try to find the resolution in the text
                lines = response_text.split('\n')
                for line in lines:
                    if term in line and "refers to" in line:
                        resolved[term] = line.strip()
                        break
        
        return resolved
    
    def _calculate_resolution_confidence(self, resolved_references: Dict[str, str]) -> float:
        """Calculate confidence in reference resolution"""
        if not resolved_references:
            return 0.0
        
        # Simple confidence based on number of resolved references
        # In practice, you might use more sophisticated methods
        return min(1.0, len(resolved_references) * 0.3)
    
    def _infer_user_preferences(self, context: SessionContext) -> Dict[str, Any]:
        """Infer user preferences from conversation patterns"""
        preferences = {}
        
        if not context.conversation_history:
            return preferences
        
        # Analyze query patterns
        query_types = []
        for turn in context.conversation_history:
            query = turn.user_query.lower()
            
            if any(word in query for word in ["show", "plot", "chart", "visualize"]):
                query_types.append("visualization")
            elif any(word in query for word in ["analyze", "analysis", "correlation"]):
                query_types.append("analysis")
            elif any(word in query for word in ["summary", "overview", "describe"]):
                query_types.append("exploration")
        
        # Determine preferred interaction style
        if query_types:
            most_common = max(set(query_types), key=query_types.count)
            preferences["preferred_interaction"] = most_common
            preferences["interaction_frequency"] = {
                qtype: query_types.count(qtype) / len(query_types)
                for qtype in set(query_types)
            }
        
        # Infer detail preference based on response patterns
        avg_response_length = sum(
            len(turn.assistant_response) for turn in context.conversation_history
        ) / len(context.conversation_history)
        
        if avg_response_length > 1000:
            preferences["detail_level"] = "high"
        elif avg_response_length > 500:
            preferences["detail_level"] = "medium"
        else:
            preferences["detail_level"] = "low"
        
        return preferences
    
    def _assess_context_coherence(self, context: SessionContext) -> float:
        """Assess coherence of conversation context"""
        if len(context.conversation_history) < 2:
            return 1.0
        
        # Simple coherence based on topic continuity
        topics = []
        for turn in context.conversation_history:
            turn_topics = self._extract_topics_from_turns([turn])
            topics.extend(turn_topics)
        
        if not topics:
            return 0.5
        
        # Calculate topic diversity (lower diversity = higher coherence)
        unique_topics = len(set(topics))
        total_topics = len(topics)
        
        coherence = 1.0 - (unique_topics / max(total_topics, 1)) * 0.8
        return max(0.0, min(1.0, coherence))
    
    def _assess_context_completeness(self, context: SessionContext) -> float:
        """Assess completeness of context information"""
        completeness_factors = []
        
        # Dataset information
        completeness_factors.append(1.0 if context.dataset_info else 0.0)
        
        # Conversation history
        completeness_factors.append(min(1.0, len(context.conversation_history) / 5))
        
        # User preferences
        completeness_factors.append(1.0 if context.user_preferences else 0.0)
        
        # Cached results
        completeness_factors.append(min(1.0, len(context.cached_results) / 3))
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _assess_average_relevance(self, context: SessionContext) -> float:
        """Assess average relevance of context items"""
        if not context.conversation_history:
            return 0.0
        
        relevance_scores = [
            turn.metadata.get("relevance_score", 0.5)
            for turn in context.conversation_history
        ]
        
        return sum(relevance_scores) / len(relevance_scores)
    
    # Context management utilities
    def get_context_status(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive context status"""
        context = self.session_manager.get_session(session_id)
        
        if not context:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "status": "active" if not context.is_expired else "expired",
            "created_at": context.created_at.isoformat(),
            "last_active": context.last_active.isoformat(),
            "conversation_turns": len(context.conversation_history),
            "cached_results": len(context.cached_results),
            "user_preferences": len(context.user_preferences),
            "memory_estimate": self._estimate_context_memory(context),
            "health_assessment": asyncio.run(self._assess_context_quality(context))
        }
    
    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """Clean up expired sessions"""
        expired_count = self.session_manager.cleanup_expired_sessions()
        
        return {
            "cleaned_sessions": expired_count,
            "timestamp": datetime.now().isoformat()
        }
