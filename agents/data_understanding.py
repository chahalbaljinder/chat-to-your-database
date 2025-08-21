"""
Data Understanding Agent - Specialized in data structure analysis and profiling
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext, DatasetInfo
from utils.data_loader import DataProfiler
from config.agent_prompts import DATA_UNDERSTANDING_PROMPT

logger = logging.getLogger(__name__)

class DataUnderstandingAgent(BaseAgent):
    """Agent specialized in understanding and analyzing data structure"""
    
    def __init__(self):
        super().__init__(
            name="DataUnderstandingAgent",
            system_prompt=DATA_UNDERSTANDING_PROMPT
        )
    
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     dataset: Optional[pd.DataFrame] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process data understanding queries"""
        
        try:
            # Determine query intent
            intent = self._classify_intent(query.lower())
            
            if intent == "overview":
                return await self._provide_dataset_overview(context, dataset)
            elif intent == "schema":
                return await self._analyze_schema(context, dataset)
            elif intent == "quality":
                return await self._assess_data_quality(context, dataset)
            elif intent == "profile":
                return await self._profile_data(context, dataset)
            elif intent == "columns":
                return await self._analyze_columns(context, dataset, query)
            else:
                return await self._general_data_understanding(query, context, dataset)
                
        except Exception as e:
            logger.error(f"Error in DataUnderstandingAgent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error analyzing data: {str(e)}",
                    response_type="error"
                )
            }
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the data understanding query"""
        if any(word in query for word in ["overview", "summary", "about", "describe"]):
            return "overview"
        elif any(word in query for word in ["schema", "structure", "columns", "types"]):
            return "schema"
        elif any(word in query for word in ["quality", "issues", "problems", "missing"]):
            return "quality"
        elif any(word in query for word in ["profile", "statistics", "distribution"]):
            return "profile"
        elif any(word in query for word in ["column", "field", "variable"]):
            return "columns"
        else:
            return "general"
    
    async def _provide_dataset_overview(self, 
                                      context: SessionContext, 
                                      dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Provide comprehensive dataset overview"""
        
        if dataset is None and context.dataset_info is None:
            return self._no_data_response()
        
        dataset_info = context.dataset_info
        
        # Create overview prompt
        overview_prompt = f"""
        Provide a comprehensive overview of this dataset:
        
        Dataset: {dataset_info.name if dataset_info else 'Unknown'}
        Shape: {dataset_info.shape if dataset_info else 'Unknown'}
        Columns: {', '.join(dataset_info.columns) if dataset_info else 'Unknown'}
        
        Data Types: {dataset_info.dtypes if dataset_info else 'Unknown'}
        
        Summary Statistics: {dataset_info.summary_stats if dataset_info else 'Unknown'}
        
        Quality Info: {dataset_info.quality_info if dataset_info else 'Unknown'}
        
        Please provide a clear, informative overview that helps the user understand their data.
        Include insights about data types, size, potential issues, and suggestions for analysis.
        """
        
        response_text = await self.generate_response(
            overview_prompt, 
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={"intent": "overview", "dataset_name": dataset_info.name if dataset_info else None}
            )
        }
    
    async def _analyze_schema(self, 
                            context: SessionContext, 
                            dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze dataset schema and structure"""
        
        if dataset is None and context.dataset_info is None:
            return self._no_data_response()
        
        dataset_info = context.dataset_info
        
        schema_prompt = f"""
        Analyze the schema and structure of this dataset:
        
        Columns ({len(dataset_info.columns) if dataset_info else 0}):
        {self._format_column_info(dataset_info) if dataset_info else 'No column information available'}
        
        Please provide:
        1. Analysis of data types and their appropriateness
        2. Identification of potential key columns (IDs, foreign keys)
        3. Suggestions for data type improvements
        4. Assessment of column relationships and hierarchy
        5. Recommendations for data modeling
        
        Be specific and actionable in your analysis.
        """
        
        response_text = await self.generate_response(
            schema_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "intent": "schema",
                    "column_count": len(dataset_info.columns) if dataset_info else 0,
                    "data_types": list(set(dataset_info.dtypes.values())) if dataset_info else []
                }
            )
        }
    
    async def _assess_data_quality(self, 
                                 context: SessionContext, 
                                 dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Assess data quality and identify issues"""
        
        if dataset is None and context.dataset_info is None:
            return self._no_data_response()
        
        dataset_info = context.dataset_info
        quality_info = dataset_info.quality_info if dataset_info else {}
        
        quality_prompt = f"""
        Assess the data quality of this dataset:
        
        Quality Assessment:
        - Total Rows: {quality_info.get('total_rows', 'Unknown')}
        - Total Columns: {quality_info.get('total_columns', 'Unknown')}
        - Memory Usage: {quality_info.get('memory_usage', 'Unknown')}
        
        Identified Issues:
        {self._format_quality_issues(quality_info.get('issues', []))}
        
        Missing Values:
        {self._format_missing_values(dataset_info.summary_stats.get('missing_values', {}) if dataset_info else {})}
        
        Please provide:
        1. Overall data quality assessment
        2. Priority ranking of identified issues
        3. Specific recommendations for data cleaning
        4. Impact assessment of each issue
        5. Suggested next steps for data preparation
        
        Be practical and prioritize actionable recommendations.
        """
        
        response_text = await self.generate_response(
            quality_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "intent": "quality",
                    "issues_found": len(quality_info.get('issues', [])),
                    "completeness": (1 - sum(dataset_info.summary_stats.get('missing_values', {}).values()) / 
                                   (dataset_info.shape[0] * dataset_info.shape[1])) * 100 if dataset_info else 0
                }
            )
        }
    
    async def _profile_data(self, 
                          context: SessionContext, 
                          dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        if dataset is None:
            return self._no_data_response()
        
        # Generate detailed profile
        profile = DataProfiler.profile_dataset(dataset)
        
        profile_prompt = f"""
        Provide a comprehensive data profile analysis:
        
        Dataset Overview:
        {profile['overview']}
        
        Column Analysis:
        {self._format_column_profiles(profile['columns'])}
        
        Correlations:
        {profile['correlations']}
        
        Patterns Detected:
        {profile['patterns']}
        
        Please provide:
        1. Key insights from the data profile
        2. Notable patterns or anomalies
        3. Recommendations for further analysis
        4. Potential modeling approaches based on the data characteristics
        5. Data visualization suggestions
        
        Focus on actionable insights that guide analysis strategy.
        """
        
        response_text = await self.generate_response(
            profile_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "intent": "profile",
                    "profile": profile
                }
            )
        }
    
    async def _analyze_columns(self, 
                             context: SessionContext, 
                             dataset: Optional[pd.DataFrame],
                             query: str) -> Dict[str, Any]:
        """Analyze specific columns mentioned in the query"""
        
        if dataset is None and context.dataset_info is None:
            return self._no_data_response()
        
        # Extract column names from query
        available_columns = context.dataset_info.columns if context.dataset_info else []
        mentioned_columns = [col for col in available_columns if col.lower() in query.lower()]
        
        if not mentioned_columns:
            mentioned_columns = available_columns[:5]  # Show first 5 if none specified
        
        column_prompt = f"""
        Analyze the following columns in detail:
        
        Columns to analyze: {', '.join(mentioned_columns)}
        
        Available column information:
        {self._format_specific_columns(context.dataset_info, mentioned_columns) if context.dataset_info else 'No detailed column information available'}
        
        User Query: {query}
        
        Please provide:
        1. Detailed analysis of each requested column
        2. Data type appropriateness and suggestions
        3. Value distribution insights
        4. Potential relationships with other columns
        5. Recommendations for analysis or transformation
        
        Be specific to the columns mentioned in the user's query.
        """
        
        response_text = await self.generate_response(
            column_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "intent": "columns",
                    "analyzed_columns": mentioned_columns
                }
            )
        }
    
    async def _general_data_understanding(self, 
                                        query: str,
                                        context: SessionContext, 
                                        dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Handle general data understanding queries"""
        
        context_info = "No dataset loaded" if context.dataset_info is None else f"""
        Current Dataset: {context.dataset_info.name}
        Shape: {context.dataset_info.shape}
        Columns: {', '.join(context.dataset_info.columns[:10])}{'...' if len(context.dataset_info.columns) > 10 else ''}
        """
        
        general_prompt = f"""
        Answer this data understanding question based on the current dataset:
        
        Question: {query}
        
        Current Dataset Context:
        {context_info}
        
        Please provide a helpful, informative response that addresses the user's question
        about understanding their data. If you need more specific information about the data,
        suggest what additional analysis might be helpful.
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
                response_type="text",
                metadata={"intent": "general"}
            )
        }
    
    def _no_data_response(self) -> Dict[str, Any]:
        """Response when no data is available"""
        return {
            "success": False,
            "response": AgentResponse(
                agent_name=self.name,
                content="No dataset is currently loaded. Please load a dataset first to analyze its structure and characteristics.",
                response_type="info"
            )
        }
    
    def _format_column_info(self, dataset_info: DatasetInfo) -> str:
        """Format column information for display"""
        if not dataset_info:
            return "No column information available"
        
        lines = []
        for col in dataset_info.columns:
            dtype = dataset_info.dtypes.get(col, 'unknown')
            unique_count = dataset_info.summary_stats.get('unique_values', {}).get(col, 'unknown')
            missing_count = dataset_info.summary_stats.get('missing_values', {}).get(col, 'unknown')
            
            lines.append(f"  - {col}: {dtype} ({unique_count} unique, {missing_count} missing)")
        
        return '\n'.join(lines)
    
    def _format_quality_issues(self, issues: List[Dict[str, Any]]) -> str:
        """Format quality issues for display"""
        if not issues:
            return "No major quality issues detected"
        
        lines = []
        for issue in issues:
            lines.append(f"  - {issue['type']}: {issue['description']}")
        
        return '\n'.join(lines)
    
    def _format_missing_values(self, missing_values: Dict[str, int]) -> str:
        """Format missing values information"""
        if not missing_values:
            return "No missing value information available"
        
        lines = []
        for col, count in missing_values.items():
            if count > 0:
                lines.append(f"  - {col}: {count} missing values")
        
        return '\n'.join(lines) if lines else "No missing values detected"
    
    def _format_column_profiles(self, column_profiles: Dict[str, Dict[str, Any]]) -> str:
        """Format column profiles for display"""
        lines = []
        for col, profile in list(column_profiles.items())[:5]:  # Show first 5 columns
            lines.append(f"  {col}:")
            lines.append(f"    Type: {profile.get('dtype', 'unknown')}")
            lines.append(f"    Completeness: {profile.get('completeness', 0):.1f}%")
            lines.append(f"    Unique values: {profile.get('unique_count', 'unknown')}")
            
        if len(column_profiles) > 5:
            lines.append(f"  ... and {len(column_profiles) - 5} more columns")
        
        return '\n'.join(lines)
    
    def _format_specific_columns(self, dataset_info: DatasetInfo, columns: List[str]) -> str:
        """Format specific column information"""
        if not dataset_info:
            return "No column information available"
        
        lines = []
        for col in columns:
            if col in dataset_info.columns:
                dtype = dataset_info.dtypes.get(col, 'unknown')
                unique_count = dataset_info.summary_stats.get('unique_values', {}).get(col, 'unknown')
                missing_count = dataset_info.summary_stats.get('missing_values', {}).get(col, 'unknown')
                
                lines.append(f"  {col}:")
                lines.append(f"    Type: {dtype}")
                lines.append(f"    Unique values: {unique_count}")
                lines.append(f"    Missing values: {missing_count}")
        
        return '\n'.join(lines)
