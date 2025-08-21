"""
Data Analysis Agent - Specialized in statistical analysis and data science
"""
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Any, Optional, Tuple
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext
from config.agent_prompts import DATA_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

class DataAnalysisAgent(BaseAgent):
    """Agent specialized in statistical analysis and data science operations"""
    
    def __init__(self):
        super().__init__(
            name="DataAnalysisAgent",
            system_prompt=DATA_ANALYSIS_PROMPT
        )
    
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     dataset: Optional[pd.DataFrame] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process data analysis queries"""
        
        try:
            if dataset is None:
                return self._no_data_response()
            
            # Classify analysis type
            analysis_type = self._classify_analysis_type(query.lower())
            
            if analysis_type == "aggregation":
                return await self._aggregation_analysis(query, context, dataset)
            elif analysis_type == "data_display":
                return await self._data_display(query, context, dataset)
            elif analysis_type == "descriptive":
                return await self._descriptive_analysis(query, context, dataset)
            elif analysis_type == "correlation":
                return await self._correlation_analysis(query, context, dataset)
            elif analysis_type == "trend":
                return await self._trend_analysis(query, context, dataset)
            elif analysis_type == "comparative":
                return await self._comparative_analysis(query, context, dataset)
            elif analysis_type == "clustering":
                return await self._clustering_analysis(query, context, dataset)
            elif analysis_type == "statistical_test":
                return await self._statistical_test(query, context, dataset)
            else:
                return await self._general_analysis(query, context, dataset)
                
        except Exception as e:
            logger.error(f"Error in DataAnalysisAgent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error performing analysis: {str(e)}",
                    response_type="error"
                )
            }
    
    def _classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis requested"""
        analysis_keywords = {
            "aggregation": ["highest", "lowest", "maximum", "minimum", "best", "worst", "top", "bottom", "total", "sum", "average", "which", "rank", "ranking", "per", "by", "each"],
            "data_display": ["show", "display", "view", "get", "find", "extract", "filter", "where", "data of", "information about"],
            "descriptive": ["describe", "summary", "statistics", "mean", "median", "std", "distribution"],
            "correlation": ["correlation", "corr", "relationship", "associated", "related"],
            "trend": ["trend", "time", "over time", "pattern", "change", "growth"],
            "comparative": ["compare", "vs", "versus", "difference", "between", "group"],
            "clustering": ["cluster", "segment", "group", "similar", "kmeans"],
            "statistical_test": ["test", "hypothesis", "significant", "p-value", "t-test", "chi-square"]
        }
        
        scores = {}
        query_lower = query.lower()
        for analysis_type, keywords in analysis_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[analysis_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    async def _data_display(self, 
                           query: str, 
                           context: SessionContext, 
                           dataset: pd.DataFrame) -> Dict[str, Any]:
        """Handle data display and filtering requests"""
        
        # Extract filter criteria from query
        query_lower = query.lower()
        
        # Look for specific values/names to filter by
        import re
        
        # Find quoted strings or specific values
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', query)
        
        # Common filter patterns
        filter_value = None
        filter_column = None
        
        # Check for quoted values first
        if quoted_matches:
            filter_value = quoted_matches[0]
        else:
            # Look for common value patterns (e.g., "show slack", "data of slack")
            # Skip common stop words and get the LAST meaningful word (usually the target)
            words = query.split()
            stop_words = ['show', 'me', 'the', 'data', 'of', 'for', 'about', 'information', 'details', 'get', 'find', 'display']
            
            # Get all non-stop words
            meaningful_words = [word for word in words if word.lower() not in stop_words]
            
            # Use the last meaningful word as it's usually the target (e.g., "show me data for Slack" -> "Slack")
            if meaningful_words:
                filter_value = meaningful_words[-1]
        
        if not filter_value:
            # If no specific filter found, show first few rows
            display_data = dataset.head(10)
            result_description = f"Showing first 10 rows of the dataset ({len(dataset)} total rows):"
        else:
            # Try to find the value in any column (case-insensitive)
            matching_rows = pd.DataFrame()
            matched_columns = []
            
            for col in dataset.columns:
                if dataset[col].dtype == 'object':  # Text columns
                    # Try exact match first, then partial match
                    exact_mask = dataset[col].str.lower() == filter_value.lower()
                    if exact_mask.any():
                        matching_rows = pd.concat([matching_rows, dataset[exact_mask]])
                        matched_columns.append(col)
                    else:
                        # Only do partial match if no exact match found and value is longer than 3 chars
                        if len(filter_value) > 3:
                            mask = dataset[col].str.contains(filter_value, case=False, na=False)
                            if mask.any():
                                matching_rows = pd.concat([matching_rows, dataset[mask]])
                                matched_columns.append(col)
                elif dataset[col].dtype in ['int64', 'float64']:  # Numeric columns
                    try:
                        numeric_value = float(filter_value)
                        mask = dataset[col] == numeric_value
                        if mask.any():
                            matching_rows = pd.concat([matching_rows, dataset[mask]])
                            matched_columns.append(col)
                    except ValueError:
                        pass
            
            # Remove duplicates
            matching_rows = matching_rows.drop_duplicates()
            display_data = matching_rows
            
            if len(matching_rows) > 0:
                result_description = f"Found {len(matching_rows)} row(s) matching '{filter_value}' in column(s): {', '.join(matched_columns)}"
            else:
                result_description = f"No rows found matching '{filter_value}'. Showing first 10 rows instead:"
                display_data = dataset.head(10)
        
        # Convert data to display format
        if len(display_data) > 0:
            # Convert to dictionary for display
            data_dict = display_data.to_dict('records')
            
            # Format the data for readable display
            formatted_data = []
            for i, row in enumerate(data_dict):
                formatted_row = {}
                for col, value in row.items():
                    # Handle different data types
                    if pd.isna(value):
                        formatted_row[col] = "N/A"
                    elif isinstance(value, (int, float)):
                        formatted_row[col] = str(value)
                    else:
                        formatted_row[col] = str(value)
                formatted_data.append(formatted_row)
            
            # Create response content
            if filter_value and len(matching_rows) > 0:
                response_content = f"## {result_description}\n\n"
                
                # Display the matching data in a formatted way
                for i, row in enumerate(formatted_data):
                    response_content += f"**Row {i+1}:**\n"
                    for col, value in row.items():
                        response_content += f"- **{col}**: {value}\n"
                    response_content += "\n"
                
                if len(formatted_data) > 5:
                    response_content += f"... and {len(formatted_data) - 5} more rows\n"
                    
            else:
                # Show in table format for general display
                response_content = f"## {result_description}\n\n"
                
                if len(formatted_data) > 0:
                    # Create a simple table format
                    headers = list(formatted_data[0].keys())
                    response_content += "| " + " | ".join(headers) + " |\n"
                    response_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    
                    for row in formatted_data[:10]:  # Limit to 10 rows for readability
                        values = [str(row.get(col, "N/A")) for col in headers]
                        response_content += "| " + " | ".join(values) + " |\n"
                    
                    if len(formatted_data) > 10:
                        response_content += f"\n*Showing first 10 rows out of {len(formatted_data)} total matching rows*\n"
        else:
            response_content = "No data to display."
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_content,
                response_type="data_display",
                metadata={
                    "analysis_type": "data_display",
                    "rows_displayed": len(display_data),
                    "total_dataset_rows": len(dataset),
                    "filter_value": filter_value,
                    "matched_columns": matched_columns if 'matched_columns' in locals() else []
                },
                artifacts={"displayed_data": formatted_data if 'formatted_data' in locals() else []}
            )
        }
    
    async def _descriptive_analysis(self, 
                                  query: str, 
                                  context: SessionContext, 
                                  dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        
        # Generate comprehensive descriptive statistics
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        
        analysis_results = {}
        
        # Numeric analysis
        if len(numeric_cols) > 0:
            numeric_desc = dataset[numeric_cols].describe()
            
            # Additional statistics
            analysis_results["numeric_summary"] = {
                "basic_stats": numeric_desc.to_dict(),
                "skewness": dataset[numeric_cols].skew().to_dict(),
                "kurtosis": dataset[numeric_cols].kurtosis().to_dict(),
                "outliers": self._detect_outliers(dataset[numeric_cols])
            }
        
        # Categorical analysis  
        if len(categorical_cols) > 0:
            analysis_results["categorical_summary"] = {}
            for col in categorical_cols:
                value_counts = dataset[col].value_counts()
                analysis_results["categorical_summary"][col] = {
                    "unique_count": dataset[col].nunique(),
                    "most_frequent": value_counts.head(5).to_dict(),
                    "missing_count": dataset[col].isnull().sum()
                }
        
        # Missing values analysis
        missing_analysis = dataset.isnull().sum()
        analysis_results["missing_values"] = missing_analysis[missing_analysis > 0].to_dict()
        
        # Generate insights
        insights_prompt = f"""
        Provide insights on this descriptive analysis:
        
        Query: {query}
        
        Numeric Statistics:
        {analysis_results.get('numeric_summary', {})}
        
        Categorical Summary:
        {analysis_results.get('categorical_summary', {})}
        
        Missing Values:
        {analysis_results.get('missing_values', {})}
        
        Please provide:
        1. Key statistical insights
        2. Notable patterns or anomalies
        3. Data quality observations
        4. Recommendations for further analysis
        5. Business implications if applicable
        
        Be specific and actionable in your insights.
        """
        
        response_text = await self.generate_response(
            insights_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "analysis_type": "descriptive",
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols)
                },
                artifacts={"analysis_results": analysis_results}
            )
        }
    
    async def _correlation_analysis(self, 
                                  query: str, 
                                  context: SessionContext, 
                                  dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis"""
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Correlation analysis requires at least 2 numeric columns.",
                    response_type="info"
                )
            }
        
        # Calculate correlations
        corr_matrix = dataset[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_val,
                        "strength": self._correlation_strength(abs(corr_val))
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        analysis_results = {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations[:10],  # Top 10
            "summary": {
                "total_pairs": len(strong_correlations),
                "strong_positive": len([c for c in strong_correlations if c["correlation"] > 0.7]),
                "strong_negative": len([c for c in strong_correlations if c["correlation"] < -0.7])
            }
        }
        
        correlation_prompt = f"""
        Analyze these correlation results:
        
        Query: {query}
        
        Strong Correlations Found:
        {self._format_correlations(strong_correlations[:5])}
        
        Summary Statistics:
        - Total variable pairs analyzed: {len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2}
        - Strong correlations (|r| > 0.5): {len(strong_correlations)}
        - Strong positive correlations (r > 0.7): {analysis_results['summary']['strong_positive']}
        - Strong negative correlations (r < -0.7): {analysis_results['summary']['strong_negative']}
        
        Please provide:
        1. Interpretation of the strongest correlations
        2. Potential business or domain implications
        3. Warnings about correlation vs causation
        4. Suggestions for further investigation
        5. Recommendations for predictive modeling
        
        Focus on actionable insights from the correlation patterns.
        """
        
        response_text = await self.generate_response(
            correlation_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "analysis_type": "correlation",
                    "variables_analyzed": len(numeric_cols),
                    "strong_correlations_found": len(strong_correlations)
                },
                artifacts={"analysis_results": analysis_results}
            )
        }
    
    async def _trend_analysis(self, 
                            query: str, 
                            context: SessionContext, 
                            dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis"""
        
        # Try to identify time columns
        time_columns = []
        for col in dataset.columns:
            if dataset[col].dtype in ['datetime64[ns]', 'datetime64']:
                time_columns.append(col)
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(dataset[col])
                    time_columns.append(col)
                except:
                    pass
        
        if not time_columns:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No time-based columns found for trend analysis. Please ensure your dataset has date/time columns.",
                    response_type="info"
                )
            }
        
        # Use first time column found
        time_col = time_columns[0]
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No numeric columns found for trend analysis.",
                    response_type="info"
                )
            }
        
        # Sort by time column
        df_sorted = dataset.sort_values(time_col)
        
        # Calculate trends for numeric columns
        trend_results = {}
        for col in numeric_cols:
            if df_sorted[col].notna().sum() > 10:  # Need sufficient data points
                # Simple linear trend
                x = range(len(df_sorted))
                y = df_sorted[col].dropna()
                x_clean = [i for i, v in enumerate(df_sorted[col]) if pd.notna(v)]
                
                if len(x_clean) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y)
                    trend_results[col] = {
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                        "significance": "significant" if p_value < 0.05 else "not significant"
                    }
        
        analysis_results = {
            "time_column": time_col,
            "trend_analysis": trend_results,
            "time_range": {
                "start": df_sorted[time_col].min(),
                "end": df_sorted[time_col].max(),
                "span": str(df_sorted[time_col].max() - df_sorted[time_col].min())
            }
        }
        
        trend_prompt = f"""
        Analyze these trend analysis results:
        
        Query: {query}
        
        Time Period: {analysis_results['time_range']['start']} to {analysis_results['time_range']['end']}
        Span: {analysis_results['time_range']['span']}
        
        Trend Analysis Results:
        {self._format_trend_results(trend_results)}
        
        Please provide:
        1. Interpretation of significant trends
        2. Business implications of the trends
        3. Seasonal or cyclical patterns if apparent
        4. Forecast implications and recommendations
        5. Suggestions for deeper time series analysis
        
        Focus on actionable insights from the temporal patterns.
        """
        
        response_text = await self.generate_response(
            trend_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "analysis_type": "trend",
                    "time_column": time_col,
                    "variables_analyzed": len(trend_results)
                },
                artifacts={"analysis_results": analysis_results}
            )
        }
    
    async def _comparative_analysis(self, 
                                  query: str, 
                                  context: SessionContext, 
                                  dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform comparative analysis between groups"""
        
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        # For text-only datasets, perform qualitative comparison
        if len(categorical_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No categorical data available for comparison.",
                    response_type="info"
                )
            }
        
        # Extract specific items to compare from the query
        import re
        
        # Look for specific items to compare (e.g., "Slack vs Discord")
        vs_pattern = r'(\w+)\s+vs\s+(\w+)'
        compare_match = re.search(vs_pattern, query, re.IGNORECASE)
        
        if compare_match:
            item1, item2 = compare_match.groups()
            
            # Find rows matching these items
            matching_rows = pd.DataFrame()
            for col in categorical_cols:
                mask1 = dataset[col].str.contains(item1, case=False, na=False)
                mask2 = dataset[col].str.contains(item2, case=False, na=False)
                matching_rows = pd.concat([matching_rows, dataset[mask1 | mask2]])
            
            matching_rows = matching_rows.drop_duplicates()
            
            if len(matching_rows) >= 2:
                # Perform qualitative comparison
                comparison_data = matching_rows.to_dict('records')
                
                comparison_prompt = f"""
                Compare {item1} vs {item2} based on the following data:
                
                Query: {query}
                
                Data for comparison:
                {json.dumps(comparison_data, indent=2)}
                
                Please provide:
                1. Direct comparison of {item1} vs {item2}
                2. Key differences in features and capabilities
                3. Strengths and weaknesses of each
                4. Use cases where each would be better
                5. Recommendations based on different scenarios
                
                Focus on practical insights and actionable recommendations.
                """
                
                response_text = await self.generate_response(
                    comparison_prompt,
                    context.get_recent_context()
                )
                
                return {
                    "success": True,
                    "response": AgentResponse(
                        agent_name=self.name,
                        content=response_text,
                        response_type="analysis",
                        metadata={
                            "analysis_type": "comparative",
                            "items_compared": [item1, item2],
                            "comparison_type": "qualitative"
                        },
                        artifacts={"comparison_data": comparison_data}
                    )
                }
        
        # If no numeric columns, still try to do qualitative analysis
        if len(numeric_cols) == 0:
            # Use first categorical column as grouping variable for qualitative analysis
            group_col = categorical_cols[0]
            groups = dataset[group_col].unique()
            
            if len(groups) > 10:
                top_groups = dataset[group_col].value_counts().head(5).index
                groups = top_groups
            
            # Create qualitative comparison
            group_data = {}
            for group in groups:
                group_rows = dataset[dataset[group_col] == group]
                group_data[str(group)] = group_rows.to_dict('records')[0] if len(group_rows) > 0 else {}
            
            comparison_prompt = f"""
            Analyze and compare these different groups/categories:
            
            Query: {query}
            
            Grouping by: {group_col}
            
            Group Data:
            {json.dumps(group_data, indent=2)}
            
            Please provide:
            1. Key characteristics of each group
            2. Main differences between groups
            3. Relative advantages and disadvantages
            4. Recommendations for different use cases
            5. Insights about the overall patterns
            
            Focus on practical comparisons and actionable insights.
            """
            
            response_text = await self.generate_response(
                comparison_prompt,
                context.get_recent_context()
            )
            
            return {
                "success": True,
                "response": AgentResponse(
                    agent_name=self.name,
                    content=response_text,
                    response_type="analysis",
                    metadata={
                        "analysis_type": "comparative",
                        "grouping_variable": group_col,
                        "groups_count": len(groups),
                        "comparison_type": "qualitative"
                    },
                    artifacts={"group_data": group_data}
                )
            }
        
        # Original numeric comparison logic for datasets with numeric columns
        group_col = categorical_cols[0]
        groups = dataset[group_col].unique()
        
        if len(groups) > 10:
            top_groups = dataset[group_col].value_counts().head(5).index
            groups = top_groups
        
        comparison_results = {}
        
        for num_col in numeric_cols:
            group_stats = {}
            for group in groups:
                group_data = dataset[dataset[group_col] == group][num_col].dropna()
                if len(group_data) > 0:
                    group_stats[str(group)] = {
                        "count": len(group_data),
                        "mean": group_data.mean(),
                        "median": group_data.median(),
                        "std": group_data.std(),
                        "min": group_data.min(),
                        "max": group_data.max()
                    }
            
            comparison_results[num_col] = group_stats
        
        # Perform statistical tests if appropriate
        statistical_tests = {}
        if len(groups) == 2:
            for num_col in numeric_cols:
                group1_data = dataset[dataset[group_col] == groups[0]][num_col].dropna()
                group2_data = dataset[dataset[group_col] == groups[1]][num_col].dropna()
                
                if len(group1_data) > 5 and len(group2_data) > 5:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    statistical_tests[num_col] = {
                        "test": "independent t-test",
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
        
        analysis_results = {
            "grouping_variable": group_col,
            "groups_analyzed": list(groups),
            "comparison_results": comparison_results,
            "statistical_tests": statistical_tests
        }
        
        comparison_prompt = f"""
        Analyze these comparative analysis results:
        
        Query: {query}
        
        Grouping Variable: {group_col}
        Groups Compared: {', '.join(map(str, groups))}
        
        Comparison Results:
        {self._format_comparison_results(comparison_results, groups)}
        
        Statistical Tests:
        {self._format_statistical_tests(statistical_tests)}
        
        Please provide:
        1. Key differences between groups
        2. Statistical significance of differences
        3. Business implications of the comparisons
        4. Recommendations based on group performance
        5. Suggestions for further analysis or action
        
        Focus on actionable insights from the group comparisons.
        """
        
        response_text = await self.generate_response(
            comparison_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "analysis_type": "comparative",
                    "grouping_variable": group_col,
                    "groups_count": len(groups)
                },
                artifacts={"analysis_results": analysis_results}
            )
        }
    
    async def _aggregation_analysis(self, 
                                   query: str, 
                                   context: SessionContext, 
                                   dataset: pd.DataFrame) -> Dict[str, Any]:
        """Handle aggregation and ranking queries like 'highest', 'lowest', 'best', 'total', etc."""
        
        query_lower = query.lower()
        
        # Detect aggregation type
        agg_type = None
        if any(word in query_lower for word in ['highest', 'maximum', 'max', 'best', 'top']):
            agg_type = 'max'
        elif any(word in query_lower for word in ['lowest', 'minimum', 'min', 'worst', 'bottom']):
            agg_type = 'min'
        elif any(word in query_lower for word in ['total', 'sum']):
            agg_type = 'sum'
        elif any(word in query_lower for word in ['average', 'mean']):
            agg_type = 'mean'
        
        # Detect what to aggregate by (grouping column)
        group_by_col = None
        target_col = None
        
        # Common grouping patterns
        if 'day' in query_lower or 'date' in query_lower:
            group_by_col = 'Date'
        elif 'line' in query_lower or 'production line' in query_lower:
            group_by_col = 'Production_Line'
        elif 'shift' in query_lower:
            group_by_col = 'Shift'
        
        # Detect target column to analyze
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if 'efficiency' in query_lower or 'oee' in query_lower:
            target_col = 'OEE_Percentage'
        elif 'production' in query_lower and 'units' in query_lower:
            target_col = 'Units_Produced'
        elif 'quality' in query_lower:
            target_col = 'Quality_Score'
        elif 'downtime' in query_lower:
            target_col = 'Downtime_Minutes'
        elif 'defects' in query_lower:
            target_col = 'Defects'
        elif 'temperature' in query_lower:
            target_col = 'Temperature'
        elif 'humidity' in query_lower:
            target_col = 'Humidity'
        
        # If we couldn't detect specifics, try to infer from available columns
        if not target_col and len(numeric_cols) > 0:
            # For efficiency queries, default to OEE if available
            if 'efficiency' in query_lower and 'OEE_Percentage' in numeric_cols:
                target_col = 'OEE_Percentage'
            else:
                target_col = numeric_cols[0]  # Default to first numeric column
        
        if not target_col:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Could not identify the metric to analyze. Please specify what you'd like to calculate (e.g., efficiency, production, quality).",
                    response_type="error"
                )
            }
        
        try:
            # Perform the aggregation
            if group_by_col and group_by_col in dataset.columns:
                # Group by the specified column
                grouped_data = dataset.groupby(group_by_col)[target_col].agg(['sum', 'mean', 'max', 'min', 'count']).round(2)
                
                # Get the result based on aggregation type
                if agg_type == 'max':
                    result_value = grouped_data['max'].max()
                    result_group = grouped_data['max'].idxmax()
                    operation = f"highest {target_col.replace('_', ' ').lower()}"
                elif agg_type == 'min':
                    result_value = grouped_data['min'].min()
                    result_group = grouped_data['min'].idxmin()
                    operation = f"lowest {target_col.replace('_', ' ').lower()}"
                elif agg_type == 'sum':
                    result_value = grouped_data['sum'].max()
                    result_group = grouped_data['sum'].idxmax()
                    operation = f"total {target_col.replace('_', ' ').lower()}"
                elif agg_type == 'mean':
                    result_value = grouped_data['mean'].max()
                    result_group = grouped_data['mean'].idxmax()
                    operation = f"average {target_col.replace('_', ' ').lower()}"
                else:
                    # Default to mean for overall efficiency
                    result_value = grouped_data['mean'].max()
                    result_group = grouped_data['mean'].idxmax()
                    operation = f"best {target_col.replace('_', ' ').lower()}"
                
                # Create detailed response
                response_content = f"## Analysis Result: {operation.title()}\n\n"
                response_content += f"**Answer:** {result_group} had the {operation} with a value of **{result_value}**\n\n"
                response_content += f"### Detailed Breakdown by {group_by_col.replace('_', ' ')}:\n\n"
                
                # Create table
                response_content += f"| {group_by_col.replace('_', ' ')} | Average | Maximum | Minimum | Total | Count |\n"
                response_content += "|" + "---|" * 6 + "\n"
                
                for group_name, row in grouped_data.iterrows():
                    response_content += f"| {group_name} | {row['mean']:.1f} | {row['max']:.1f} | {row['min']:.1f} | {row['sum']:.1f} | {row['count']} |\n"
                
                # Add insights
                response_content += f"\n### Key Insights:\n"
                response_content += f"- **Winner:** {result_group} with {result_value} {target_col.replace('_', ' ').lower()}\n"
                
                # Find runner-up
                sorted_groups = grouped_data.sort_values('mean', ascending=False)
                if len(sorted_groups) > 1:
                    runner_up = sorted_groups.index[1]
                    runner_up_value = sorted_groups.iloc[1]['mean']
                    response_content += f"- **Runner-up:** {runner_up} with {runner_up_value} average {target_col.replace('_', ' ').lower()}\n"
                
                # Calculate performance difference
                if len(sorted_groups) > 1:
                    best_avg = sorted_groups.iloc[0]['mean']
                    worst_avg = sorted_groups.iloc[-1]['mean']
                    diff_pct = ((best_avg - worst_avg) / worst_avg * 100)
                    response_content += f"- **Performance Gap:** {diff_pct:.1f}% difference between best and worst performing {group_by_col.replace('_', ' ').lower()}\n"
                
            else:
                # Simple aggregation without grouping
                if agg_type == 'max':
                    result_value = dataset[target_col].max()
                    operation = f"maximum {target_col.replace('_', ' ').lower()}"
                elif agg_type == 'min':
                    result_value = dataset[target_col].min()
                    operation = f"minimum {target_col.replace('_', ' ').lower()}"
                elif agg_type == 'sum':
                    result_value = dataset[target_col].sum()
                    operation = f"total {target_col.replace('_', ' ').lower()}"
                else:
                    result_value = dataset[target_col].mean()
                    operation = f"average {target_col.replace('_', ' ').lower()}"
                
                response_content = f"## Analysis Result\n\n"
                response_content += f"**The {operation} is: {result_value:.2f}**\n\n"
                response_content += f"### Additional Statistics for {target_col.replace('_', ' ')}:\n"
                response_content += f"- **Mean:** {dataset[target_col].mean():.2f}\n"
                response_content += f"- **Maximum:** {dataset[target_col].max():.2f}\n"
                response_content += f"- **Minimum:** {dataset[target_col].min():.2f}\n"
                response_content += f"- **Standard Deviation:** {dataset[target_col].std():.2f}\n"
            
            return {
                "success": True,
                "response": AgentResponse(
                    agent_name=self.name,
                    content=response_content,
                    response_type="analysis",
                    metadata={"analysis_type": "aggregation", "target_column": target_col, "group_by": group_by_col}
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error performing aggregation analysis: {str(e)}",
                    response_type="error"
                )
            }
    
    async def _clustering_analysis(self, 
                                 query: str, 
                                 context: SessionContext, 
                                 dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Clustering analysis requires at least 2 numeric columns.",
                    response_type="info"
                )
            }
        
        # Prepare data
        cluster_data = dataset[numeric_cols].dropna()
        
        if len(cluster_data) < 10:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Insufficient data points for clustering analysis (need at least 10).",
                    response_type="info"
                )
            }
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, min(11, len(cluster_data) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find optimal k (simplified)
        optimal_k = k_range[0]  # Default to 2
        if len(inertias) >= 3:
            # Find the elbow point
            for i in range(1, len(inertias) - 1):
                if inertias[i-1] - inertias[i] > (inertias[i] - inertias[i+1]) * 1.5:
                    optimal_k = k_range[i]
                    break
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        cluster_data_with_labels = cluster_data.copy()
        cluster_data_with_labels['cluster'] = cluster_labels
        
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_subset = cluster_data_with_labels[cluster_data_with_labels['cluster'] == cluster_id]
            cluster_analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_subset),
                "percentage": len(cluster_subset) / len(cluster_data_with_labels) * 100,
                "characteristics": cluster_subset[numeric_cols].mean().to_dict()
            }
        
        analysis_results = {
            "optimal_clusters": optimal_k,
            "cluster_analysis": cluster_analysis,
            "features_used": list(numeric_cols),
            "total_data_points": len(cluster_data)
        }
        
        clustering_prompt = f"""
        Analyze these clustering results:
        
        Query: {query}
        
        Clustering Summary:
        - Optimal number of clusters: {optimal_k}
        - Features used: {', '.join(numeric_cols)}
        - Total data points clustered: {len(cluster_data)}
        
        Cluster Characteristics:
        {self._format_cluster_results(cluster_analysis)}
        
        Please provide:
        1. Interpretation of each cluster's characteristics
        2. Business meaning of the clusters (customer segments, product groups, etc.)
        3. Recommendations for cluster-specific strategies
        4. Insights about data patterns revealed by clustering
        5. Suggestions for using clusters in decision-making
        
        Focus on actionable business insights from the clustering.
        """
        
        response_text = await self.generate_response(
            clustering_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={
                    "analysis_type": "clustering",
                    "optimal_clusters": optimal_k,
                    "features_used": len(numeric_cols)
                },
                artifacts={"analysis_results": analysis_results}
            )
        }
    
    async def _statistical_test(self, 
                              query: str, 
                              context: SessionContext, 
                              dataset: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical hypothesis tests"""
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated test selection
        
        statistical_prompt = f"""
        The user is asking for statistical testing:
        
        Query: {query}
        
        Available data:
        - Numeric columns: {len(dataset.select_dtypes(include=[np.number]).columns)}
        - Categorical columns: {len(dataset.select_dtypes(include=['object', 'category']).columns)}
        - Sample size: {len(dataset)}
        
        Please explain:
        1. What type of statistical test would be appropriate
        2. What assumptions need to be checked
        3. How to interpret the results
        4. Limitations of the proposed test
        5. Alternative approaches if needed
        
        Provide guidance on statistical testing strategy.
        """
        
        response_text = await self.generate_response(
            statistical_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="analysis",
                metadata={"analysis_type": "statistical_test"}
            )
        }
    
    async def _general_analysis(self, 
                              query: str, 
                              context: SessionContext, 
                              dataset: pd.DataFrame) -> Dict[str, Any]:
        """Handle general analysis requests with actual data insights"""
        
        # Provide actual data overview instead of just guidance
        data_overview = {
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "data_types": dataset.dtypes.to_dict(),
            "numeric_columns": list(dataset.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
            "missing_values": dataset.isnull().sum().to_dict(),
            "sample_data": dataset.head(3).to_dict('records')
        }
        
        # Generate basic insights about the data
        insights = []
        
        # Dataset size insight
        insights.append(f"Dataset contains {data_overview['shape'][0]} rows and {data_overview['shape'][1]} columns")
        
        # Missing values insight
        missing_cols = [col for col, count in data_overview['missing_values'].items() if count > 0]
        if missing_cols:
            insights.append(f"Missing values found in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}")
        else:
            insights.append("No missing values detected")
        
        # Column types insight
        if data_overview['numeric_columns']:
            insights.append(f"Numeric columns available for analysis: {', '.join(data_overview['numeric_columns'][:3])}")
        if data_overview['categorical_columns']:
            insights.append(f"Categorical columns for grouping: {', '.join(data_overview['categorical_columns'][:3])}")
        
        # Create formatted response
        response_content = f"## Dataset Overview\n\n"
        
        # Basic info
        response_content += f"**Size:** {data_overview['shape'][0]} rows × {data_overview['shape'][1]} columns\n\n"
        
        # Column information
        response_content += "**Columns:**\n"
        for col in data_overview['columns']:
            dtype = str(data_overview['data_types'][col])
            missing = data_overview['missing_values'][col]
            response_content += f"- **{col}** ({dtype})"
            if missing > 0:
                response_content += f" - {missing} missing values"
            response_content += "\n"
        
        response_content += "\n"
        
        # Sample data
        response_content += "**Sample Data (first 3 rows):**\n\n"
        if data_overview['sample_data']:
            headers = list(data_overview['sample_data'][0].keys())
            response_content += "| " + " | ".join(headers) + " |\n"
            response_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in data_overview['sample_data']:
                values = [str(row.get(col, "N/A")) for col in headers]
                response_content += "| " + " | ".join(values) + " |\n"
        
        response_content += "\n**Key Insights:**\n"
        for insight in insights:
            response_content += f"- {insight}\n"
        
        response_content += f"""
**What you can ask me:**
- "Show me the data for [specific value]" - Filter and display specific records
- "Describe [column name]" - Get statistical summary of a column  
- "Compare [column1] vs [column2]" - Analyze relationships between columns
- "Show me correlations" - Find relationships between numeric variables
- "Create a chart of [data]" - Generate visualizations
"""
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_content,
                response_type="overview",
                metadata={
                    "analysis_type": "general_overview",
                    "dataset_shape": data_overview['shape'],
                    "numeric_columns": len(data_overview['numeric_columns']),
                    "categorical_columns": len(data_overview['categorical_columns'])
                },
                artifacts={"data_overview": data_overview}
            )
        }
    
    # Helper methods for formatting results
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100),
                "bounds": {"lower": float(lower), "upper": float(upper)}
            }
        
        return outliers
    
    def _correlation_strength(self, corr_value: float) -> str:
        """Classify correlation strength"""
        if corr_value >= 0.7:
            return "strong"
        elif corr_value >= 0.3:
            return "moderate"
        else:
            return "weak"
    
    def _format_correlations(self, correlations: List[Dict]) -> str:
        """Format correlation results for display"""
        lines = []
        for corr in correlations:
            lines.append(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']})")
        return '\n'.join(lines)
    
    def _format_trend_results(self, trends: Dict[str, Dict]) -> str:
        """Format trend analysis results"""
        lines = []
        for var, trend in trends.items():
            direction = trend['trend_direction']
            significance = trend['significance']
            r2 = trend['r_squared']
            lines.append(f"  {var}: {direction} trend (R² = {r2:.3f}, {significance})")
        return '\n'.join(lines)
    
    def _format_comparison_results(self, results: Dict, groups: List) -> str:
        """Format comparison results"""
        lines = []
        for var, group_stats in results.items():
            lines.append(f"  {var}:")
            for group in groups:
                if str(group) in group_stats:
                    stats = group_stats[str(group)]
                    lines.append(f"    {group}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        return '\n'.join(lines)
    
    def _format_statistical_tests(self, tests: Dict) -> str:
        """Format statistical test results"""
        if not tests:
            return "No statistical tests performed"
        
        lines = []
        for var, test in tests.items():
            sig_text = "significant" if test['significant'] else "not significant"
            lines.append(f"  {var}: {test['test']} (p = {test['p_value']:.4f}, {sig_text})")
        return '\n'.join(lines)
    
    def _format_cluster_results(self, clusters: Dict) -> str:
        """Format clustering results"""
        lines = []
        for cluster_id, info in clusters.items():
            lines.append(f"  {cluster_id.replace('_', ' ').title()}:")
            lines.append(f"    Size: {info['size']} ({info['percentage']:.1f}%)")
            lines.append(f"    Key characteristics: {list(info['characteristics'].keys())[:3]}")
        return '\n'.join(lines)
    
    def _no_data_response(self) -> Dict[str, Any]:
        """Response when no data is available"""
        return {
            "success": False,
            "response": AgentResponse(
                agent_name=self.name,
                content="No dataset is currently loaded. Please load a dataset first to perform analysis.",
                response_type="info"
            )
        }
