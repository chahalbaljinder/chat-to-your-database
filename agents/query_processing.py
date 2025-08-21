"""
Query Processing Agent - Specialized in interpreting natural language queries
"""
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext
from config.agent_prompts import QUERY_PROCESSING_PROMPT

logger = logging.getLogger(__name__)

class QueryProcessingAgent(BaseAgent):
    """Agent specialized in processing and interpreting natural language queries"""
    
    def __init__(self):
        super().__init__(
            name="QueryProcessingAgent", 
            system_prompt=QUERY_PROCESSING_PROMPT
        )
        self.query_patterns = self._init_query_patterns()
    
    def _init_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize query patterns for intent classification"""
        return {
            "filter": ["show", "get", "find", "where", "filter", "select"],
            "aggregate": ["sum", "count", "average", "mean", "max", "min", "group", "total"],
            "sort": ["sort", "order", "arrange", "rank", "top", "bottom"],
            "compare": ["compare", "versus", "vs", "difference", "between"],
            "correlation": ["correlation", "corr", "relationship", "relate", "associated"],
            "visualization": ["plot", "chart", "graph", "show", "visualize", "display"],
            "statistics": ["statistics", "stats", "describe", "summary", "distribution"]
        }
    
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     dataset: Optional[pd.DataFrame] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process natural language query and generate appropriate code/instructions"""
        
        try:
            # Analyze the query
            analysis = await self._analyze_query(query, context)
            
            # Generate response based on analysis
            if analysis["intent"] == "code_generation":
                return await self._generate_code(query, analysis, context, dataset)
            elif analysis["intent"] == "reference_resolution":
                return await self._resolve_references(query, analysis, context)
            elif analysis["intent"] == "multi_step":
                return await self._handle_multi_step_query(query, analysis, context)
            else:
                return await self._general_query_processing(query, analysis, context)
                
        except Exception as e:
            logger.error(f"Error in QueryProcessingAgent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error processing query: {str(e)}",
                    response_type="error"
                )
            }
    
    async def _analyze_query(self, query: str, context: SessionContext) -> Dict[str, Any]:
        """Analyze the query to understand intent and extract information"""
        
        query_lower = query.lower()
        
        # Extract entities (column names, values, etc.)
        entities = self._extract_entities(query, context)
        
        # Classify intent
        intent_scores = {}
        for intent, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "general"
        
        # Check for references to previous results
        has_references = self._check_for_references(query_lower)
        
        # Determine complexity
        complexity = self._assess_complexity(query, entities)
        
        analysis = {
            "original_query": query,
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "entities": entities,
            "has_references": has_references,
            "complexity": complexity,
            "intent": self._determine_processing_intent(primary_intent, has_references, complexity)
        }
        
        return analysis
    
    def _extract_entities(self, query: str, context: SessionContext) -> Dict[str, List[str]]:
        """Extract entities like column names, values, etc. from the query"""
        entities = {
            "columns": [],
            "values": [],
            "operators": [],
            "aggregations": []
        }
        
        if context.dataset_info and context.dataset_info.columns:
            # Find column names mentioned in the query
            for col in context.dataset_info.columns:
                if col.lower() in query.lower():
                    entities["columns"].append(col)
        
        # Extract quoted values
        quoted_values = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
        entities["values"].extend(quoted_values)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities["values"].extend(numbers)
        
        # Find operators
        operators = ["=", "!=", ">", "<", ">=", "<=", "like", "in", "not in"]
        for op in operators:
            if op in query.lower():
                entities["operators"].append(op)
        
        # Find aggregation functions
        agg_functions = ["sum", "count", "avg", "mean", "max", "min", "std", "median"]
        for agg in agg_functions:
            if agg in query.lower():
                entities["aggregations"].append(agg)
        
        return entities
    
    def _check_for_references(self, query: str) -> List[str]:
        """Check for references to previous analyses"""
        reference_patterns = [
            "that", "it", "this", "previous", "last", "earlier", "above", "before",
            "same", "similar", "like that", "from before"
        ]
        
        found_references = []
        for pattern in reference_patterns:
            if pattern in query:
                found_references.append(pattern)
        
        return found_references
    
    def _assess_complexity(self, query: str, entities: Dict[str, List[str]]) -> str:
        """Assess query complexity"""
        complexity_indicators = [
            len(entities["columns"]) > 2,
            len(entities["aggregations"]) > 1,
            "and" in query.lower() or "or" in query.lower(),
            "group by" in query.lower(),
            "join" in query.lower(),
            len(query.split()) > 15
        ]
        
        complexity_score = sum(complexity_indicators)
        
        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _determine_processing_intent(self, primary_intent: str, has_references: List[str], complexity: str) -> str:
        """Determine how to process the query"""
        if has_references:
            return "reference_resolution"
        elif complexity == "high":
            return "multi_step"
        elif primary_intent in ["filter", "aggregate", "sort"]:
            return "code_generation"
        else:
            return "general"
    
    async def _generate_code(self, 
                           query: str, 
                           analysis: Dict[str, Any], 
                           context: SessionContext,
                           dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Generate pandas/SQL code for the query"""
        
        if not context.dataset_info:
            return self._no_data_response()
        
        code_prompt = f"""
        Generate Python pandas code to answer this query:
        
        Query: {query}
        
        Available Dataset:
        - Name: {context.dataset_info.name}
        - Columns: {', '.join(context.dataset_info.columns)}
        - Data types: {analysis['entities']['columns']}
        
        Query Analysis:
        - Intent: {analysis['primary_intent']}
        - Entities found: {analysis['entities']}
        - Complexity: {analysis['complexity']}
        
        Requirements:
        1. Generate clean, readable pandas code
        2. Use the variable name 'df' for the dataframe
        3. Handle potential errors (missing columns, wrong data types)
        4. Include comments explaining the operations
        5. Provide the result in a format suitable for further analysis
        
        Only return the executable Python code, no markdown formatting.
        """
        
        response_text = await self.generate_response(
            code_prompt,
            context.get_recent_context()
        )
        
        # Clean up the response to extract just the code
        code = self._extract_code_from_response(response_text)
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="code",
                metadata={
                    "intent": analysis["primary_intent"],
                    "code": code,
                    "entities": analysis["entities"]
                },
                artifacts={"generated_code": code}
            )
        }
    
    async def _resolve_references(self, 
                                query: str, 
                                analysis: Dict[str, Any], 
                                context: SessionContext) -> Dict[str, Any]:
        """Resolve references to previous analyses"""
        
        # Get recent conversation history
        recent_context = context.get_recent_context()
        recent_responses = []
        
        if context.conversation_history:
            recent_responses = [
                {
                    "query": turn.user_query,
                    "response": turn.assistant_response,
                    "metadata": turn.metadata
                }
                for turn in context.conversation_history[-3:]  # Last 3 turns
            ]
        
        resolution_prompt = f"""
        Resolve references in this query based on recent conversation:
        
        Current Query: {query}
        References found: {analysis['has_references']}
        
        Recent Conversation:
        {self._format_recent_responses(recent_responses)}
        
        Current Dataset Context:
        - Columns: {', '.join(context.dataset_info.columns) if context.dataset_info else 'No dataset loaded'}
        
        Please:
        1. Identify what "that", "it", "this", etc. refer to in the context
        2. Rewrite the query with resolved references
        3. Explain what was resolved
        4. Suggest how to proceed with the clarified query
        
        Provide a clear interpretation of what the user is asking for.
        """
        
        response_text = await self.generate_response(
            resolution_prompt,
            recent_context
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="interpretation",
                metadata={
                    "original_query": query,
                    "references": analysis["has_references"],
                    "resolved": True
                }
            )
        }
    
    async def _handle_multi_step_query(self, 
                                     query: str, 
                                     analysis: Dict[str, Any], 
                                     context: SessionContext) -> Dict[str, Any]:
        """Break down complex queries into steps"""
        
        multi_step_prompt = f"""
        Break down this complex query into manageable steps:
        
        Query: {query}
        
        Query Analysis:
        - Primary Intent: {analysis['primary_intent']}
        - Entities: {analysis['entities']}
        - Complexity: {analysis['complexity']}
        
        Available Dataset Context:
        {self._format_dataset_context(context.dataset_info) if context.dataset_info else 'No dataset loaded'}
        
        Please:
        1. Break the query into 3-5 logical steps
        2. Explain what each step accomplishes
        3. Identify which agent should handle each step
        4. Suggest the order of execution
        5. Highlight any dependencies between steps
        
        Focus on creating a clear workflow that can be executed systematically.
        """
        
        response_text = await self.generate_response(
            multi_step_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="workflow",
                metadata={
                    "original_query": query,
                    "complexity": analysis["complexity"],
                    "multi_step": True
                }
            )
        }
    
    async def _general_query_processing(self, 
                                      query: str, 
                                      analysis: Dict[str, Any], 
                                      context: SessionContext) -> Dict[str, Any]:
        """Handle general query processing"""
        
        general_prompt = f"""
        Process and interpret this data query:
        
        Query: {query}
        
        Analysis Results:
        - Primary Intent: {analysis['primary_intent']}
        - Entities Found: {analysis['entities']}
        - Complexity: {analysis['complexity']}
        
        Dataset Context:
        {self._format_dataset_context(context.dataset_info) if context.dataset_info else 'No dataset available'}
        
        Please:
        1. Interpret what the user wants to accomplish
        2. Identify the best approach to answer their question
        3. Suggest which type of analysis would be most appropriate
        4. Recommend next steps or clarifying questions if needed
        
        Provide a helpful interpretation that guides the analysis process.
        """
        
        response_text = await self.generate_response(
            general_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="interpretation",
                metadata={
                    "intent": analysis["primary_intent"],
                    "entities": analysis["entities"]
                }
            )
        }
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from the agent response"""
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code without markdown
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('df') or line.strip().startswith('import') or in_code:
                code_lines.append(line)
                in_code = True
            elif in_code and (not line.strip() or line.startswith(' ')):
                code_lines.append(line)
            elif in_code:
                break
        
        return '\n'.join(code_lines).strip()
    
    def _format_recent_responses(self, responses: List[Dict[str, Any]]) -> str:
        """Format recent responses for context"""
        if not responses:
            return "No recent conversation history"
        
        formatted = []
        for i, resp in enumerate(responses, 1):
            formatted.append(f"Turn {i}:")
            formatted.append(f"  User: {resp['query']}")
            formatted.append(f"  Assistant: {resp['response'][:200]}...")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def _format_dataset_context(self, dataset_info) -> str:
        """Format dataset context for prompts"""
        if not dataset_info:
            return "No dataset information available"
        
        return f"""
        Dataset: {dataset_info.name}
        Shape: {dataset_info.shape}
        Columns: {', '.join(dataset_info.columns)}
        Data Types: {len(set(dataset_info.dtypes.values()))} unique types
        """
    
    def _no_data_response(self) -> Dict[str, Any]:
        """Response when no data is available"""
        return {
            "success": False,
            "response": AgentResponse(
                agent_name=self.name,
                content="No dataset is currently loaded. Please load a dataset first to process queries about data.",
                response_type="info"
            )
        }
