"""
Insight Generation Agent - Specialized in generating business intelligence and actionable insights
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext
from config.agent_prompts import INSIGHT_GENERATION_PROMPT

logger = logging.getLogger(__name__)

class InsightGenerationAgent(BaseAgent):
    """Agent specialized in generating business intelligence and actionable insights"""
    
    def __init__(self):
        super().__init__(
            name="InsightGenerationAgent",
            system_prompt=INSIGHT_GENERATION_PROMPT
        )
    
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     dataset: Optional[pd.DataFrame] = None,
                     analysis_results: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process insight generation requests"""
        
        try:
            # Classify insight type
            insight_type = self._classify_insight_type(query.lower())
            
            if insight_type == "business_intelligence":
                return await self._generate_business_intelligence(query, context, dataset, analysis_results)
            elif insight_type == "pattern_analysis":
                return await self._analyze_patterns(query, context, dataset, analysis_results)
            elif insight_type == "recommendations":
                return await self._generate_recommendations(query, context, dataset, analysis_results)
            elif insight_type == "anomaly_detection":
                return await self._detect_anomalies(query, context, dataset, analysis_results)
            elif insight_type == "predictive_insights":
                return await self._generate_predictive_insights(query, context, dataset, analysis_results)
            elif insight_type == "summary":
                return await self._generate_summary_insights(query, context, dataset, analysis_results)
            else:
                return await self._generate_general_insights(query, context, dataset, analysis_results)
                
        except Exception as e:
            logger.error(f"Error in InsightGenerationAgent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error generating insights: {str(e)}",
                    response_type="error"
                )
            }
    
    def _classify_insight_type(self, query: str) -> str:
        """Classify the type of insight requested"""
        insight_keywords = {
            "business_intelligence": ["business", "intelligence", "strategy", "performance", "kpi", "metrics"],
            "pattern_analysis": ["pattern", "trend", "behavior", "analysis", "discover", "identify"],
            "recommendations": ["recommend", "suggest", "what should", "action", "next steps", "improve"],
            "anomaly_detection": ["anomaly", "outlier", "unusual", "abnormal", "strange", "irregular"],
            "predictive_insights": ["predict", "forecast", "future", "expect", "will", "projection"],
            "summary": ["summary", "overview", "key findings", "main insights", "conclude", "wrap up"]
        }
        
        scores = {}
        for insight_type, keywords in insight_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                scores[insight_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    async def _generate_business_intelligence(self, 
                                            query: str, 
                                            context: SessionContext, 
                                            dataset: Optional[pd.DataFrame],
                                            analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate business intelligence insights"""
        
        # Gather context from the session
        session_summary = self._create_session_summary(context)
        data_overview = self._create_data_overview(dataset) if dataset is not None else "No dataset available"
        
        bi_prompt = f"""
        Generate business intelligence insights based on the conversation and analysis:
        
        User Query: {query}
        
        Session Context:
        {session_summary}
        
        Data Overview:
        {data_overview}
        
        Analysis Results:
        {self._format_analysis_results(analysis_results) if analysis_results else "No specific analysis results provided"}
        
        Please provide:
        1. Key business insights from the data and analysis
        2. Performance indicators and their implications
        3. Strategic recommendations based on findings
        4. Risk factors and opportunities identified
        5. Competitive advantages or disadvantages revealed
        6. Action items for stakeholders
        
        Focus on strategic, business-relevant insights that drive decision-making.
        """
        
        response_text = await self.generate_response(
            bi_prompt,
            context.get_recent_context()
        )
        
        # Extract key metrics if dataset is available
        key_metrics = {}
        if dataset is not None:
            key_metrics = self._calculate_key_metrics(dataset)
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="business_intelligence",
                metadata={
                    "insight_type": "business_intelligence",
                    "session_turns": len(context.conversation_history)
                },
                artifacts={
                    "key_metrics": key_metrics,
                    "session_summary": session_summary
                }
            )
        }
    
    async def _analyze_patterns(self, 
                              query: str, 
                              context: SessionContext, 
                              dataset: Optional[pd.DataFrame],
                              analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in the data and conversation"""
        
        patterns_found = []
        
        if dataset is not None:
            patterns_found.extend(self._detect_data_patterns(dataset))
        
        if analysis_results:
            patterns_found.extend(self._extract_patterns_from_analysis(analysis_results))
        
        conversation_patterns = self._analyze_conversation_patterns(context)
        
        pattern_prompt = f"""
        Analyze and interpret patterns discovered in the data and analysis:
        
        User Query: {query}
        
        Data Patterns Discovered:
        {self._format_patterns(patterns_found)}
        
        Conversation Patterns:
        {conversation_patterns}
        
        Analysis Context:
        {self._format_analysis_results(analysis_results) if analysis_results else "No specific analysis results"}
        
        Please provide:
        1. Interpretation of significant patterns
        2. Hidden relationships or connections revealed
        3. Recurring themes or behaviors
        4. Seasonal or cyclical patterns if applicable
        5. Implications of pattern combinations
        6. Actionable insights from pattern analysis
        
        Focus on patterns that provide strategic value and actionable insights.
        """
        
        response_text = await self.generate_response(
            pattern_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="pattern_analysis",
                metadata={
                    "insight_type": "pattern_analysis",
                    "patterns_found": len(patterns_found)
                },
                artifacts={
                    "patterns": patterns_found,
                    "conversation_patterns": conversation_patterns
                }
            )
        }
    
    async def _generate_recommendations(self, 
                                      query: str, 
                                      context: SessionContext, 
                                      dataset: Optional[pd.DataFrame],
                                      analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        
        # Gather insights from the session
        key_findings = self._extract_key_findings(context, analysis_results)
        data_insights = self._extract_data_insights(dataset) if dataset is not None else {}
        
        recommendation_prompt = f"""
        Generate specific, actionable recommendations based on the analysis and conversation:
        
        User Query: {query}
        
        Key Findings from Analysis:
        {key_findings}
        
        Data Insights:
        {data_insights}
        
        Session Context:
        {self._create_session_summary(context)}
        
        Please provide:
        1. Immediate action items (short-term, 1-30 days)
        2. Strategic recommendations (medium-term, 1-6 months)
        3. Long-term strategic initiatives (6+ months)
        4. Resource requirements and priorities
        5. Risk mitigation strategies
        6. Success metrics to track progress
        7. Alternative approaches or contingency plans
        
        Make recommendations specific, measurable, and achievable. Prioritize by impact and feasibility.
        """
        
        response_text = await self.generate_response(
            recommendation_prompt,
            context.get_recent_context()
        )
        
        # Structure recommendations
        structured_recommendations = self._structure_recommendations(response_text)
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="recommendations",
                metadata={
                    "insight_type": "recommendations",
                    "recommendation_categories": len(structured_recommendations)
                },
                artifacts={
                    "structured_recommendations": structured_recommendations,
                    "key_findings": key_findings
                }
            )
        }
    
    async def _detect_anomalies(self, 
                              query: str, 
                              context: SessionContext, 
                              dataset: Optional[pd.DataFrame],
                              analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect and analyze anomalies"""
        
        anomalies_found = []
        
        if dataset is not None:
            anomalies_found.extend(self._detect_data_anomalies(dataset))
        
        if analysis_results:
            anomalies_found.extend(self._extract_anomalies_from_analysis(analysis_results))
        
        anomaly_prompt = f"""
        Analyze anomalies and unusual patterns discovered:
        
        User Query: {query}
        
        Anomalies Detected:
        {self._format_anomalies(anomalies_found)}
        
        Data Context:
        {self._create_data_overview(dataset) if dataset is not None else "No dataset available"}
        
        Please provide:
        1. Interpretation of each anomaly
        2. Potential causes or explanations
        3. Business impact assessment
        4. Risk evaluation (opportunity vs. threat)
        5. Investigation priorities
        6. Monitoring recommendations
        7. Preventive or corrective actions
        
        Focus on anomalies that could significantly impact business operations or outcomes.
        """
        
        response_text = await self.generate_response(
            anomaly_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="anomaly_analysis",
                metadata={
                    "insight_type": "anomaly_detection",
                    "anomalies_found": len(anomalies_found)
                },
                artifacts={"anomalies": anomalies_found}
            )
        }
    
    async def _generate_predictive_insights(self, 
                                          query: str, 
                                          context: SessionContext, 
                                          dataset: Optional[pd.DataFrame],
                                          analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictive insights and forecasts"""
        
        predictive_indicators = []
        
        if dataset is not None:
            predictive_indicators.extend(self._identify_predictive_indicators(dataset))
        
        if analysis_results:
            predictive_indicators.extend(self._extract_predictive_signals(analysis_results))
        
        predictive_prompt = f"""
        Generate predictive insights and forecasts based on the analysis:
        
        User Query: {query}
        
        Predictive Indicators:
        {self._format_predictive_indicators(predictive_indicators)}
        
        Trend Analysis:
        {self._extract_trends_from_context(context)}
        
        Data Characteristics:
        {self._create_data_overview(dataset) if dataset is not None else "No dataset available"}
        
        Please provide:
        1. Likely future scenarios based on current trends
        2. Leading indicators to monitor
        3. Potential inflection points or changes
        4. Confidence levels for predictions
        5. Risk factors that could change outcomes
        6. Recommended forecasting approaches
        7. Data collection needs for better predictions
        
        Base predictions on solid analytical foundations and acknowledge uncertainties.
        """
        
        response_text = await self.generate_response(
            predictive_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="predictive_insights",
                metadata={
                    "insight_type": "predictive_insights",
                    "predictive_indicators": len(predictive_indicators)
                },
                artifacts={"predictive_indicators": predictive_indicators}
            )
        }
    
    async def _generate_summary_insights(self, 
                                       query: str, 
                                       context: SessionContext, 
                                       dataset: Optional[pd.DataFrame],
                                       analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary insights from the entire session"""
        
        session_insights = self._synthesize_session_insights(context)
        key_discoveries = self._extract_key_discoveries(context, analysis_results)
        
        summary_prompt = f"""
        Provide a comprehensive summary of insights from this analysis session:
        
        User Query: {query}
        
        Session Overview:
        - Total conversation turns: {len(context.conversation_history)}
        - Dataset analyzed: {context.dataset_info.name if context.dataset_info else "None"}
        - Analysis types performed: {self._get_analysis_types(context)}
        
        Key Discoveries:
        {key_discoveries}
        
        Session Insights:
        {session_insights}
        
        Please provide:
        1. Executive summary of key findings
        2. Most significant insights discovered
        3. Connection between different analyses
        4. Strategic implications for the business
        5. Unanswered questions for future investigation
        6. Overall assessment of data quality and completeness
        7. Recommended next steps for continued analysis
        
        Create a compelling narrative that connects all insights into a coherent story.
        """
        
        response_text = await self.generate_response(
            summary_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="summary_insights",
                metadata={
                    "insight_type": "summary",
                    "session_length": len(context.conversation_history)
                },
                artifacts={
                    "session_insights": session_insights,
                    "key_discoveries": key_discoveries
                }
            )
        }
    
    async def _generate_general_insights(self, 
                                       query: str, 
                                       context: SessionContext, 
                                       dataset: Optional[pd.DataFrame],
                                       analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate general insights"""
        
        general_prompt = f"""
        Generate insights for this request:
        
        User Query: {query}
        
        Available Context:
        {self._create_comprehensive_context(context, dataset, analysis_results)}
        
        Please provide valuable insights that address the user's question and help them understand their data better.
        Focus on actionable information that can guide decision-making.
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
                response_type="general_insights",
                metadata={"insight_type": "general"}
            )
        }
    
    # Helper methods for insight generation
    def _create_session_summary(self, context: SessionContext) -> str:
        """Create a summary of the current session"""
        if not context.conversation_history:
            return "No conversation history available"
        
        summary_parts = [
            f"Session Duration: {len(context.conversation_history)} conversation turns",
            f"Dataset: {context.dataset_info.name if context.dataset_info else 'None loaded'}"
        ]
        
        # Extract common themes
        themes = []
        for turn in context.conversation_history:
            if "analysis" in turn.user_query.lower():
                themes.append("analysis")
            if "visualization" in turn.user_query.lower() or "plot" in turn.user_query.lower():
                themes.append("visualization")
            if "correlation" in turn.user_query.lower():
                themes.append("correlation")
        
        if themes:
            unique_themes = list(set(themes))
            summary_parts.append(f"Main topics discussed: {', '.join(unique_themes)}")
        
        return '\n'.join(summary_parts)
    
    def _create_data_overview(self, dataset: pd.DataFrame) -> str:
        """Create an overview of the dataset"""
        if dataset is None:
            return "No dataset available"
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        
        overview = [
            f"Dataset shape: {dataset.shape}",
            f"Numeric columns: {len(numeric_cols)}",
            f"Categorical columns: {len(categorical_cols)}",
            f"Missing values: {dataset.isnull().sum().sum()}",
            f"Memory usage: {dataset.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
        ]
        
        return '\n'.join(overview)
    
    def _format_analysis_results(self, analysis_results: Dict[str, Any]) -> str:
        """Format analysis results for prompt inclusion"""
        if not analysis_results:
            return "No analysis results provided"
        
        formatted_parts = []
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                formatted_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, list):
                formatted_parts.append(f"{key}: {len(value)} items")
            else:
                formatted_parts.append(f"{key}: {str(value)[:100]}")
        
        return '\n'.join(formatted_parts)
    
    def _calculate_key_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key metrics from the dataset"""
        metrics = {}
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metrics["numeric_summary"] = {
                col: {
                    "mean": float(dataset[col].mean()),
                    "median": float(dataset[col].median()),
                    "std": float(dataset[col].std())
                }
                for col in numeric_cols[:5]  # Limit to first 5
            }
        
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            metrics["categorical_summary"] = {
                col: {
                    "unique_count": int(dataset[col].nunique()),
                    "most_frequent": str(dataset[col].mode().iloc[0]) if len(dataset[col].mode()) > 0 else "N/A"
                }
                for col in categorical_cols[:3]  # Limit to first 3
            }
        
        return metrics
    
    def _detect_data_patterns(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect patterns in the dataset"""
        patterns = []
        
        # High correlation patterns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = dataset[numeric_cols].corr()
            high_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corrs.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": corr_val
                        })
            
            if high_corrs:
                patterns.append({
                    "type": "high_correlation",
                    "description": f"Found {len(high_corrs)} high correlations",
                    "details": high_corrs[:5]  # Top 5
                })
        
        # Missing value patterns
        missing_counts = dataset.isnull().sum()
        high_missing = missing_counts[missing_counts > len(dataset) * 0.5]
        if len(high_missing) > 0:
            patterns.append({
                "type": "high_missing_values",
                "description": f"{len(high_missing)} columns with >50% missing values",
                "details": high_missing.to_dict()
            })
        
        return patterns
    
    def _extract_patterns_from_analysis(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from analysis results"""
        patterns = []
        
        if "strong_correlations" in analysis_results:
            correlations = analysis_results["strong_correlations"]
            if len(correlations) > 0:
                patterns.append({
                    "type": "correlation_patterns",
                    "description": f"Found {len(correlations)} strong correlations in analysis",
                    "details": correlations[:3]
                })
        
        if "cluster_analysis" in analysis_results:
            clusters = analysis_results["cluster_analysis"]
            patterns.append({
                "type": "clustering_patterns",
                "description": f"Data segments into {len(clusters)} distinct clusters",
                "details": clusters
            })
        
        return patterns
    
    def _analyze_conversation_patterns(self, context: SessionContext) -> str:
        """Analyze patterns in the conversation"""
        if not context.conversation_history:
            return "No conversation history to analyze"
        
        query_types = []
        for turn in context.conversation_history:
            query = turn.user_query.lower()
            if any(word in query for word in ["show", "plot", "chart", "visualize"]):
                query_types.append("visualization")
            elif any(word in query for word in ["analyze", "correlation", "relationship"]):
                query_types.append("analysis")
            elif any(word in query for word in ["describe", "summary", "overview"]):
                query_types.append("exploration")
            else:
                query_types.append("other")
        
        type_counts = {qtype: query_types.count(qtype) for qtype in set(query_types)}
        most_common = max(type_counts, key=type_counts.get)
        
        return f"Conversation focused primarily on {most_common} ({type_counts[most_common]} queries). Types: {type_counts}"
    
    def _extract_key_findings(self, context: SessionContext, analysis_results: Optional[Dict[str, Any]]) -> str:
        """Extract key findings from context and analysis"""
        findings = []
        
        if context.conversation_history:
            # Look for insights in recent responses
            recent_responses = context.conversation_history[-3:]
            for turn in recent_responses:
                if "significant" in turn.assistant_response.lower() or "important" in turn.assistant_response.lower():
                    findings.append(f"From {turn.user_query[:50]}...: Key insight found")
        
        if analysis_results:
            for key, value in analysis_results.items():
                if "correlation" in key.lower() and isinstance(value, list) and len(value) > 0:
                    findings.append(f"Strong correlations identified in {key}")
                elif "trend" in key.lower():
                    findings.append(f"Trend patterns found in {key}")
        
        return '\n'.join(findings) if findings else "No specific key findings identified"
    
    def _extract_data_insights(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Extract insights from dataset characteristics"""
        if dataset is None:
            return {}
        
        insights = {
            "size": f"Dataset contains {len(dataset):,} rows and {len(dataset.columns)} columns",
            "completeness": f"Data is {(1 - dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100:.1f}% complete"
        }
        
        # Data balance insights
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:2]:  # Check first 2 categorical columns
                value_counts = dataset[col].value_counts()
                if len(value_counts) > 1:
                    balance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    if balance_ratio > 10:
                        insights[f"{col}_balance"] = f"Highly imbalanced - top category is {balance_ratio:.1f}x more frequent than least frequent"
        
        return insights
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for display"""
        if not patterns:
            return "No significant patterns detected"
        
        formatted = []
        for pattern in patterns:
            formatted.append(f"- {pattern['type']}: {pattern['description']}")
        
        return '\n'.join(formatted)
    
    def _format_anomalies(self, anomalies: List[Dict[str, Any]]) -> str:
        """Format anomalies for display"""
        if not anomalies:
            return "No significant anomalies detected"
        
        formatted = []
        for anomaly in anomalies:
            formatted.append(f"- {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        
        return '\n'.join(formatted)
    
    def _detect_data_anomalies(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the dataset"""
        anomalies = []
        
        # Outliers in numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = dataset[(dataset[col] < Q1 - 1.5 * IQR) | (dataset[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                anomalies.append({
                    "type": "outliers",
                    "description": f"{len(outliers)} outliers detected in {col}",
                    "column": col,
                    "count": len(outliers)
                })
        
        return anomalies
    
    def _extract_anomalies_from_analysis(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract anomalies from analysis results"""
        anomalies = []
        
        if "outliers" in analysis_results:
            outlier_info = analysis_results["outliers"]
            for col, info in outlier_info.items():
                if info.get("count", 0) > 0:
                    anomalies.append({
                        "type": "statistical_outliers",
                        "description": f"{info['count']} outliers in {col}",
                        "column": col
                    })
        
        return anomalies
    
    def _structure_recommendations(self, response_text: str) -> Dict[str, List[str]]:
        """Structure recommendations into categories"""
        # Simple parsing - in practice, you might use more sophisticated NLP
        structured = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }
        
        lines = response_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if "immediate" in line.lower() or "short" in line.lower():
                current_category = "immediate"
            elif "strategic" in line.lower() or "medium" in line.lower():
                current_category = "short_term"
            elif "long" in line.lower():
                current_category = "long_term"
            elif line.startswith('-') or line.startswith('•') and current_category:
                structured[current_category].append(line)
        
        return structured
    
    def _identify_predictive_indicators(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential predictive indicators"""
        indicators = []
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        # Look for trending variables (simplified)
        for col in numeric_cols:
            if len(dataset) > 10:
                # Simple trend detection
                values = dataset[col].dropna()
                if len(values) > 10:
                    x = range(len(values))
                    trend_strength = np.corrcoef(x, values)[0, 1]
                    
                    if abs(trend_strength) > 0.3:
                        indicators.append({
                            "type": "trend_indicator",
                            "column": col,
                            "trend_strength": trend_strength,
                            "description": f"{col} shows {'positive' if trend_strength > 0 else 'negative'} trend"
                        })
        
        return indicators
    
    def _extract_predictive_signals(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract predictive signals from analysis results"""
        signals = []
        
        if "trend_analysis" in analysis_results:
            trends = analysis_results["trend_analysis"]
            for var, trend_info in trends.items():
                if trend_info.get("significance") == "significant":
                    signals.append({
                        "type": "significant_trend",
                        "variable": var,
                        "direction": trend_info.get("trend_direction"),
                        "description": f"Significant trend in {var}"
                    })
        
        return signals
    
    def _format_predictive_indicators(self, indicators: List[Dict[str, Any]]) -> str:
        """Format predictive indicators"""
        if not indicators:
            return "No predictive indicators identified"
        
        formatted = []
        for indicator in indicators:
            formatted.append(f"- {indicator['type']}: {indicator['description']}")
        
        return '\n'.join(formatted)
    
    def _synthesize_session_insights(self, context: SessionContext) -> str:
        """Synthesize insights from the entire session"""
        if not context.conversation_history:
            return "No session insights available"
        
        insights = []
        
        # Analyze progression of queries
        query_evolution = []
        for turn in context.conversation_history:
            query_evolution.append(turn.user_query[:50] + "...")
        
        insights.append(f"Analysis progression: {' → '.join(query_evolution[-3:])}")
        
        # Look for patterns in responses
        response_themes = []
        for turn in context.conversation_history:
            if "correlation" in turn.assistant_response.lower():
                response_themes.append("correlation analysis")
            if "trend" in turn.assistant_response.lower():
                response_themes.append("trend analysis")
            if "distribution" in turn.assistant_response.lower():
                response_themes.append("distribution analysis")
        
        unique_themes = list(set(response_themes))
        if unique_themes:
            insights.append(f"Analysis themes covered: {', '.join(unique_themes)}")
        
        return '\n'.join(insights)
    
    def _extract_key_discoveries(self, context: SessionContext, analysis_results: Optional[Dict[str, Any]]) -> str:
        """Extract key discoveries from the session"""
        discoveries = []
        
        if context.conversation_history:
            for turn in context.conversation_history:
                # Look for discoveries in responses
                response_lower = turn.assistant_response.lower()
                if "discovered" in response_lower or "found" in response_lower:
                    # Extract the sentence containing the discovery
                    sentences = turn.assistant_response.split('.')
                    for sentence in sentences:
                        if "discovered" in sentence.lower() or "found" in sentence.lower():
                            discoveries.append(sentence.strip())
                            break
        
        if analysis_results:
            if "strong_correlations" in analysis_results and len(analysis_results["strong_correlations"]) > 0:
                discoveries.append(f"Strong correlations discovered between variables")
            if "cluster_analysis" in analysis_results:
                discoveries.append("Data segmentation patterns identified through clustering")
        
        return '\n'.join(discoveries) if discoveries else "No specific discoveries documented"
    
    def _get_analysis_types(self, context: SessionContext) -> str:
        """Get types of analysis performed in the session"""
        analysis_types = set()
        
        for turn in context.conversation_history:
            if any(agent in turn.metadata.get("agent_responses", []) for agent in ["DataAnalysisAgent"]):
                analysis_types.add("statistical analysis")
            if any(agent in turn.metadata.get("agent_responses", []) for agent in ["VisualizationAgent"]):
                analysis_types.add("visualization")
            if any(agent in turn.metadata.get("agent_responses", []) for agent in ["DataUnderstandingAgent"]):
                analysis_types.add("data profiling")
        
        return ', '.join(analysis_types) if analysis_types else "general exploration"
    
    def _extract_trends_from_context(self, context: SessionContext) -> str:
        """Extract trend information from context"""
        trends = []
        
        for turn in context.conversation_history:
            if "trend" in turn.assistant_response.lower():
                # Extract trend information
                response_parts = turn.assistant_response.split('.')
                for part in response_parts:
                    if "trend" in part.lower():
                        trends.append(part.strip())
                        break
        
        return '\n'.join(trends) if trends else "No specific trend information available"
    
    def _create_comprehensive_context(self, 
                                    context: SessionContext, 
                                    dataset: Optional[pd.DataFrame],
                                    analysis_results: Optional[Dict[str, Any]]) -> str:
        """Create comprehensive context for general insights"""
        context_parts = []
        
        # Session context
        context_parts.append(f"Session: {len(context.conversation_history)} turns")
        
        # Dataset context
        if dataset is not None:
            context_parts.append(self._create_data_overview(dataset))
        else:
            context_parts.append("No dataset loaded")
        
        # Analysis context
        if analysis_results:
            context_parts.append(self._format_analysis_results(analysis_results))
        
        # Recent conversation
        if context.conversation_history:
            recent_queries = [turn.user_query for turn in context.conversation_history[-3:]]
            context_parts.append(f"Recent queries: {'; '.join(recent_queries)}")
        
        return '\n\n'.join(context_parts)
