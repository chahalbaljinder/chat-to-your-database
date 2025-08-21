"""
Orchestrator Agent - Master controller for the multi-agent data chat system
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import re

# Import agents
from agents.base_agent import BaseAgent, AgentResponse
from agents.data_understanding import DataUnderstandingAgent
from agents.query_processing import QueryProcessingAgent
from agents.data_analysis import DataAnalysisAgent
from agents.visualization import VisualizationAgent
from agents.insight_generation import InsightGenerationAgent

# Import utilities
from utils.session_utils import SessionContext, ConversationTurn, session_manager
from config.agent_prompts import ORCHESTRATOR_PROMPT

logger = logging.getLogger(__name__)

class Orchestrator(BaseAgent):
    """
    Master orchestrator that coordinates all specialized agents and manages workflows
    """
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            system_prompt=ORCHESTRATOR_PROMPT
        )
        
        # Initialize specialized agents
        self.agents = {
            "data_understanding": DataUnderstandingAgent(),
            "query_processing": QueryProcessingAgent(),
            "data_analysis": DataAnalysisAgent(),
            "visualization": VisualizationAgent(),
            "insight_generation": InsightGenerationAgent()
        }
        
        # Intent patterns for routing
        self.intent_patterns = {
            "greeting": [
                r"\b(hello|hi|hey|greetings)\b",
                r"\bwhat can you (do|help)\b",
                r"\bintroduce\b",
                r"\bstart\b"
            ],
            "data_understanding": [
                r"\b(structure|schema|columns|shape|info|describe|overview)\b",
                r"\bwhat.*(data|dataset|file)\b",
                r"\bmissing values?\b",
                r"\bdata types?\b",
                r"\bdata quality\b"
            ],
            "query_processing": [
                r"\bfilter\b",
                r"\bselect\b",
                r"\bwhere\b",
                r"\bgroup by\b",
                r"\bshow me.*where\b"
            ],
            "data_analysis": [
                r"\b(correlation|corr|relationship)\b",
                r"\b(statistics|stats|statistical|mean|median|std)\b",
                r"\b(trend|time series|over time)\b",
                r"\b(compare|comparison|vs|versus)\b",
                r"\b(cluster|clustering|segment)\b",
                r"\b(hypothesis|test|significance)\b"
            ],
            "visualization": [
                r"\b(plot|chart|graph|visualize|visualization)\b",
                r"\b(bar chart|histogram|scatter|line chart|heatmap)\b",
                r"\b(show.*chart|create.*plot)\b",
                r"\bdashboard\b"
            ],
            "insight_generation": [
                r"\b(insight|insights|recommend|recommendation)\b",
                r"\bwhat.*(patterns|trends)\b",
                r"\bbusiness intelligence\b",
                r"\bkey findings\b",
                r"\bactionable\b",
                r"\bsummarize.*findings\b"
            ]
        }
    
    async def process(self, 
                     query: str, 
                     context: SessionContext,
                     **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract process method from BaseAgent
        """
        return await self.process_query(query, context, **kwargs)

    async def process_query(self, 
                          query: str, 
                          context: SessionContext,
                          **kwargs) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        """
        try:
            logger.info(f"Processing query for session {context.session_id}: {query[:100]}...")
            
            # Classify intent and determine agent routing
            intent = self._classify_intent(query, context)
            workflow = self._determine_workflow(intent, query, context)
            
            # Execute the workflow
            result = await self._execute_workflow(workflow, query, context)
            
            # Update session context
            if result.get("success"):
                self._update_session_context(query, result, context)
            
            # Update session in manager
            session_manager.update_session(context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"I encountered an error processing your request: {str(e)}",
                    response_type="error"
                )
            }
    
    def _classify_intent(self, query: str, context: SessionContext) -> str:
        """
        Classify the intent of the user query
        """
        query_lower = query.lower()
        
        # Check for test mode patterns
        if query.startswith("[TEST_"):
            test_match = re.search(r"\[TEST_(\w+)\]", query)
            if test_match:
                return test_match.group(1).lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score
        
        # Check context-specific intents
        if not context.dataset_info:
            # No data loaded - prioritize greeting and data understanding
            if "hello" in query_lower or "what can you" in query_lower:
                return "greeting"
            return "greeting"  # Default to greeting when no data
        
        # Find highest scoring intent
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                best_intent = max(intent_scores, key=intent_scores.get)
                return best_intent
        
        # Default classification based on context
        if "?" in query and not context.dataset_info:
            return "greeting"
        elif context.dataset_info:
            return "data_analysis"  # Default to analysis if data is loaded
        else:
            return "greeting"
    
    def _determine_workflow(self, 
                          intent: str, 
                          query: str, 
                          context: SessionContext) -> List[Dict[str, Any]]:
        """
        Determine the workflow of agents needed for the query
        """
        workflows = {
            "greeting": [
                {"agent": "orchestrator", "operation": "greeting"}
            ],
            "data_understanding": [
                {"agent": "data_understanding", "operation": "analyze"}
            ],
            "query_processing": [
                {"agent": "query_processing", "operation": "process"},
                {"agent": "data_analysis", "operation": "execute"}
            ],
            "data_analysis": [
                {"agent": "data_analysis", "operation": "analyze"}
            ],
            "visualization": [
                {"agent": "visualization", "operation": "create"}
            ],
            "insight_generation": [
                {"agent": "insight_generation", "operation": "generate"}
            ]
        }
        
        # Get base workflow
        workflow = workflows.get(intent, [{"agent": "orchestrator", "operation": "general"}])
        
        # Enhance workflow based on query complexity
        if self._is_complex_query(query):
            # For complex queries, add insight generation at the end
            if not any(step["agent"] == "insight_generation" for step in workflow):
                workflow.append({"agent": "insight_generation", "operation": "synthesize"})
        
        # If visualization terms are present, ensure visualization step
        if re.search(r"\b(plot|chart|graph|visualize)\b", query.lower()) and intent != "visualization":
            workflow.append({"agent": "visualization", "operation": "create"})
        
        return workflow
    
    async def _execute_workflow(self, 
                               workflow: List[Dict[str, Any]], 
                               query: str, 
                               context: SessionContext) -> Dict[str, Any]:
        """
        Execute the determined workflow
        """
        results = []
        agent_responses = []
        final_response = None
        
        for step in workflow:
            agent_name = step["agent"]
            operation = step.get("operation", "process")
            
            try:
                if agent_name == "orchestrator":
                    # Handle orchestrator-specific operations
                    result = await self._handle_orchestrator_operation(operation, query, context)
                else:
                    # Execute agent operation
                    agent = self.agents.get(agent_name)
                    if not agent:
                        logger.warning(f"Agent {agent_name} not found")
                        continue
                    
                    # Get dataset if available
                    dataset = None
                    if context.dataset_info:
                        try:
                            from utils.data_loader import DataLoader
                            import os
                            
                            # Try to reload the dataset from the file path
                            file_path = context.dataset_info.file_path
                            if file_path and os.path.exists(file_path):
                                dataset, _ = DataLoader.load_file(file_path)
                            else:
                                logger.warning(f"Dataset file not found: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load dataset: {str(e)}")
                    
                    result = await agent.process(query, context, dataset=dataset, operation=operation)
                
                results.append(result)
                
                if result.get("success"):
                    agent_response = result.get("response")
                    if agent_response:
                        agent_responses.append({
                            "agent": agent_name,
                            "operation": operation,
                            "response": agent_response.content,
                            "response_type": agent_response.response_type,
                            "metadata": agent_response.metadata,
                            "artifacts": agent_response.artifacts
                        })
                        
                        # Use the last successful response as final response
                        final_response = agent_response
                else:
                    logger.warning(f"Agent {agent_name} operation {operation} failed: {result.get('error')}")
            
            except Exception as e:
                logger.error(f"Error executing {agent_name}.{operation}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "agent": agent_name,
                    "operation": operation
                })
        
        # Synthesize final response
        if not final_response:
            final_response = AgentResponse(
                agent_name=self.name,
                content="I'm ready to help you analyze your data. Please upload a dataset to get started, or ask me what I can do for you!",
                response_type="greeting"
            )
        
        # If multiple agents responded, create a synthesized response
        if len(agent_responses) > 1:
            synthesis_result = await self._synthesize_responses(agent_responses, query, context)
            if synthesis_result.get("success"):
                final_response = synthesis_result.get("response", final_response)
        
        return {
            "success": len([r for r in results if r.get("success")]) > 0,
            "response": final_response,
            "agent_responses": agent_responses,
            "workflow_results": results
        }
    
    async def _handle_orchestrator_operation(self, 
                                           operation: str, 
                                           query: str, 
                                           context: SessionContext) -> Dict[str, Any]:
        """
        Handle orchestrator-specific operations
        """
        if operation == "greeting":
            return await self._handle_greeting(query, context)
        elif operation == "general":
            return await self._handle_general_query(query, context)
        else:
            return {
                "success": False,
                "error": f"Unknown orchestrator operation: {operation}"
            }
    
    async def _handle_greeting(self, query: str, context: SessionContext) -> Dict[str, Any]:
        """
        Handle greeting and introductory queries
        """
        has_data = context.dataset_info is not None
        conversation_length = len(context.conversation_history)
        
        greeting_prompt = f"""
        Respond to this user query in a helpful, conversational manner:
        
        User Query: {query}
        
        Context:
        - Data loaded: {'Yes' if has_data else 'No'}
        - Conversation turns: {conversation_length}
        - Session preferences: {context.user_preferences}
        
        {"Dataset Info: " + str(context.dataset_info) if has_data else "No dataset loaded yet"}
        
        Provide a warm, helpful response that:
        1. Acknowledges the user's query
        2. Explains what you can help with
        3. Suggests next steps based on whether data is loaded
        4. Maintains a conversational, friendly tone
        
        If no data is loaded, explain how to upload data and what you can do once data is available.
        If data is loaded, suggest some analysis options.
        """
        
        response_text = await self.generate_response(greeting_prompt, {})
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="greeting",
                metadata={
                    "has_data": has_data,
                    "conversation_turns": conversation_length
                }
            )
        }
    
    async def _handle_general_query(self, query: str, context: SessionContext) -> Dict[str, Any]:
        """
        Handle general queries that don't fit specific agent categories
        """
        general_prompt = f"""
        The user has asked a general question that doesn't fit into specific analysis categories:
        
        User Query: {query}
        
        Context:
        - Data available: {'Yes' if context.dataset_info else 'No'}
        - Previous conversation: {len(context.conversation_history)} turns
        
        Provide a helpful response that:
        1. Addresses their query directly
        2. Offers relevant suggestions
        3. Guides them toward productive data analysis
        4. Maintains a conversational tone
        
        If the query seems like it could benefit from data analysis, suggest specific approaches.
        """
        
        response_text = await self.generate_response(general_prompt, context.get_recent_context())
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="general",
                metadata={"query_classification": "general"}
            )
        }
    
    async def _synthesize_responses(self, 
                                  agent_responses: List[Dict[str, Any]], 
                                  query: str, 
                                  context: SessionContext) -> Dict[str, Any]:
        """
        Synthesize responses from multiple agents into a coherent response
        """
        synthesis_prompt = f"""
        Synthesize these agent responses into a coherent, comprehensive answer:
        
        Original Query: {query}
        
        Agent Responses:
        """
        
        for i, resp in enumerate(agent_responses, 1):
            synthesis_prompt += f"\n{i}. {resp['agent'].title()} Agent ({resp['response_type']}):\n"
            synthesis_prompt += f"   {resp['response'][:300]}...\n"
        
        synthesis_prompt += """
        
        Create a unified response that:
        1. Directly answers the user's question
        2. Integrates insights from all agents
        3. Maintains logical flow and coherence
        4. Highlights key findings and recommendations
        5. Uses a conversational, helpful tone
        
        Avoid simply concatenating responses - create a synthesized, coherent answer.
        """
        
        response_text = await self.generate_response(synthesis_prompt, context.get_recent_context())
        
        # Combine artifacts from all responses
        combined_artifacts = {}
        for resp in agent_responses:
            if resp.get("artifacts"):
                combined_artifacts[f"{resp['agent']}_artifacts"] = resp["artifacts"]
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="synthesis",
                metadata={
                    "synthesized_agents": [r["agent"] for r in agent_responses],
                    "synthesis_complexity": len(agent_responses)
                },
                artifacts=combined_artifacts
            )
        }
    
    def _update_session_context(self, 
                               query: str, 
                               result: Dict[str, Any], 
                               context: SessionContext):
        """
        Update session context with query and response
        """
        response = result.get("response")
        agent_responses = result.get("agent_responses", [])
        
        # Create conversation turn
        turn = ConversationTurn(
            user_query=query,
            assistant_response=response.content if response else "No response generated",
            timestamp=datetime.now(),
            agent_responses=agent_responses,
            metadata={
                "orchestrator_workflow": True,
                "success": result.get("success", False)
            }
        )
        
        # Add to context
        context.add_conversation_turn(turn)
        
        # Cache result if it has artifacts
        if response and response.artifacts:
            cache_key = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            context.cached_results[cache_key] = {
                "query": query,
                "response": response.content,
                "artifacts": response.artifacts,
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex and might benefit from insight generation
        """
        complexity_indicators = [
            r"\b(why|how|what.*patterns|what.*trends)\b",
            r"\b(recommend|suggest|insights|actionable)\b",
            r"\b(compare.*with|relationship.*between)\b",
            r"\b(predict|forecast|future)\b",
            len(query.split()) > 10,  # Long queries
            "?" in query and len(query.split()) > 5
        ]
        
        query_lower = query.lower()
        complexity_score = 0
        
        for indicator in complexity_indicators[:-2]:  # Exclude length checks
            if re.search(indicator, query_lower):
                complexity_score += 1
        
        # Add length-based complexity
        if len(query.split()) > 10:
            complexity_score += 1
        if "?" in query and len(query.split()) > 5:
            complexity_score += 1
        
        return complexity_score >= 2
    
    # Utility methods for agent management
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about available agents"""
        return {
            "orchestrator": {
                "name": "Orchestrator",
                "description": "Master controller coordinating all agents",
                "capabilities": ["workflow_management", "intent_classification", "response_synthesis"]
            },
            **{
                name: {
                    "name": agent.name,
                    "description": f"{agent.name} specialized for specific data tasks",
                    "capabilities": ["data_processing", "analysis", "response_generation"]
                }
                for name, agent in self.agents.items()
            }
        }
    
    def get_workflow_patterns(self) -> Dict[str, List[str]]:
        """Get available workflow patterns"""
        return {
            intent: [step["agent"] for step in workflow]
            for intent, workflow in {
                "data_understanding": [{"agent": "data_understanding"}],
                "query_processing": [{"agent": "query_processing"}, {"agent": "data_analysis"}],
                "data_analysis": [{"agent": "data_analysis"}],
                "visualization": [{"agent": "visualization"}],
                "insight_generation": [{"agent": "insight_generation"}],
                "complex_analysis": [{"agent": "data_analysis"}, {"agent": "visualization"}, {"agent": "insight_generation"}]
            }.items()
        }
    
    # Health and monitoring methods
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_status = {
            "orchestrator": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        for name, agent in self.agents.items():
            try:
                # Simple health check - verify agent can initialize
                health_status[name] = "healthy" if hasattr(agent, 'process') else "error"
            except Exception as e:
                health_status[name] = f"error: {str(e)}"
        
        overall_health = "healthy" if all(
            status == "healthy" for status in health_status.values() 
            if status != health_status["timestamp"]
        ) else "degraded"
        
        health_status["overall"] = overall_health
        return health_status
