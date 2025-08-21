"""
Visualization Agent - Specialized in creating data visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from agents.base_agent import BaseAgent, AgentResponse
from utils.session_utils import SessionContext
from config.agent_prompts import VISUALIZATION_PROMPT

logger = logging.getLogger(__name__)

class VisualizationAgent(BaseAgent):
    """Agent specialized in creating data visualizations and charts"""
    
    def __init__(self):
        super().__init__(
            name="VisualizationAgent",
            system_prompt=VISUALIZATION_PROMPT
        )
        self.color_palette = px.colors.qualitative.Set1
        
    async def process(self, 
                     query: str, 
                     context: SessionContext, 
                     dataset: Optional[pd.DataFrame] = None,
                     **kwargs) -> Dict[str, Any]:
        """Process visualization requests"""
        
        try:
            if dataset is None:
                return self._no_data_response()
            
            # Classify visualization type
            viz_type = self._classify_visualization_type(query.lower())
            
            if viz_type == "distribution":
                return await self._create_distribution_plots(query, context, dataset)
            elif viz_type == "relationship":
                return await self._create_relationship_plots(query, context, dataset)
            elif viz_type == "time_series":
                return await self._create_time_series_plots(query, context, dataset)
            elif viz_type == "categorical":
                return await self._create_categorical_plots(query, context, dataset)
            elif viz_type == "comparison":
                return await self._create_comparison_plots(query, context, dataset)
            elif viz_type == "correlation":
                return await self._create_correlation_plots(query, context, dataset)
            elif viz_type == "dashboard":
                return await self._create_dashboard(query, context, dataset)
            else:
                return await self._suggest_visualization(query, context, dataset)
                
        except Exception as e:
            logger.error(f"Error in VisualizationAgent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": AgentResponse(
                    agent_name=self.name,
                    content=f"Error creating visualization: {str(e)}",
                    response_type="error"
                )
            }
    
    def _classify_visualization_type(self, query: str) -> str:
        """Classify the type of visualization requested"""
        viz_keywords = {
            "distribution": ["histogram", "distribution", "density", "box plot", "violin"],
            "relationship": ["scatter", "relationship", "correlation scatter", "bubble"],
            "time_series": ["time", "trend", "over time", "timeline", "temporal"],
            "categorical": ["bar", "count", "frequency", "pie", "donut"],
            "comparison": ["compare", "vs", "versus", "side by side"],
            "correlation": ["heatmap", "correlation matrix", "corr"],
            "dashboard": ["dashboard", "overview", "summary", "multiple", "all"]
        }
        
        scores = {}
        for viz_type, keywords in viz_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                scores[viz_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    async def _create_distribution_plots(self, 
                                       query: str, 
                                       context: SessionContext, 
                                       dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create distribution visualizations"""
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No numeric columns found for distribution plots.",
                    response_type="info"
                )
            }
        
        # Extract specific columns mentioned in query
        mentioned_columns = [col for col in numeric_cols if col.lower() in query.lower()]
        if not mentioned_columns:
            mentioned_columns = numeric_cols[:4]  # Show first 4 if none specified
        
        # Create visualizations
        visualizations = []
        
        for col in mentioned_columns:
            # Histogram
            hist_fig = px.histogram(
                dataset, 
                x=col,
                title=f'Distribution of {col}',
                nbins=30,
                marginal="box"
            )
            hist_fig.update_layout(showlegend=False)
            
            visualizations.append({
                "type": "histogram",
                "column": col,
                "title": f"Distribution of {col}",
                "figure": hist_fig.to_json()
            })
            
            # Box plot if multiple numeric columns for comparison
            if len(mentioned_columns) > 1:
                box_data = []
                for mcol in mentioned_columns:
                    values = dataset[mcol].dropna()
                    box_data.extend([{"variable": mcol, "value": val} for val in values])
                
                if box_data:
                    box_df = pd.DataFrame(box_data)
                    box_fig = px.box(
                        box_df, 
                        x="variable", 
                        y="value",
                        title="Distribution Comparison (Box Plots)"
                    )
                    
                    visualizations.append({
                        "type": "box_comparison",
                        "columns": mentioned_columns,
                        "title": "Distribution Comparison",
                        "figure": box_fig.to_json()
                    })
                    break  # Only create one comparison plot
        
        # Generate insights
        insights_prompt = f"""
        Analyze these distribution visualizations:
        
        Query: {query}
        Columns visualized: {', '.join(mentioned_columns)}
        
        Statistical summaries:
        {dataset[mentioned_columns].describe().to_dict()}
        
        Please provide:
        1. Key insights about the distributions
        2. Notable patterns (skewness, outliers, multimodality)
        3. Comparison between variables if applicable
        4. Business implications of the distributions
        5. Recommendations for further analysis
        
        Focus on actionable insights from the distribution patterns.
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
                response_type="visualization",
                metadata={
                    "viz_type": "distribution",
                    "columns_plotted": mentioned_columns,
                    "plot_count": len(visualizations)
                },
                artifacts={
                    "visualizations": visualizations,
                    "data_summary": dataset[mentioned_columns].describe().to_dict()
                }
            )
        }
    
    async def _create_relationship_plots(self, 
                                       query: str, 
                                       context: SessionContext, 
                                       dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create relationship visualizations (scatter plots, etc.)"""
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Need at least 2 numeric columns for relationship plots.",
                    response_type="info"
                )
            }
        
        # Try to extract column pairs from query
        mentioned_columns = [col for col in numeric_cols if col.lower() in query.lower()]
        
        visualizations = []
        
        if len(mentioned_columns) >= 2:
            # Create scatter plot for mentioned columns
            x_col, y_col = mentioned_columns[0], mentioned_columns[1]
            
            scatter_fig = px.scatter(
                dataset,
                x=x_col,
                y=y_col,
                title=f'{y_col} vs {x_col}',
                trendline="ols"
            )
            
            visualizations.append({
                "type": "scatter",
                "x_column": x_col,
                "y_column": y_col,
                "title": f"{y_col} vs {x_col}",
                "figure": scatter_fig.to_json()
            })
            
            # Add third column as color if available
            if len(mentioned_columns) >= 3:
                color_col = mentioned_columns[2]
                scatter_color_fig = px.scatter(
                    dataset,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f'{y_col} vs {x_col} (colored by {color_col})',
                    trendline="ols"
                )
                
                visualizations.append({
                    "type": "scatter_color",
                    "x_column": x_col,
                    "y_column": y_col,
                    "color_column": color_col,
                    "title": f"{y_col} vs {x_col} (by {color_col})",
                    "figure": scatter_color_fig.to_json()
                })
        
        else:
            # Create pairwise relationships for top correlations
            corr_matrix = dataset[numeric_cols].corr()
            
            # Find strongest correlations
            strong_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.3:  # Moderate correlation threshold
                        strong_pairs.append({
                            "x": corr_matrix.columns[i],
                            "y": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j]
                        })
            
            # Sort by absolute correlation and take top 3
            strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            for pair in strong_pairs[:3]:
                scatter_fig = px.scatter(
                    dataset,
                    x=pair["x"],
                    y=pair["y"],
                    title=f'{pair["y"]} vs {pair["x"]} (r = {pair["correlation"]:.3f})',
                    trendline="ols"
                )
                
                visualizations.append({
                    "type": "scatter",
                    "x_column": pair["x"],
                    "y_column": pair["y"],
                    "correlation": pair["correlation"],
                    "title": f'{pair["y"]} vs {pair["x"]}',
                    "figure": scatter_fig.to_json()
                })
        
        # Generate insights
        insights_prompt = f"""
        Analyze these relationship visualizations:
        
        Query: {query}
        
        Relationships plotted:
        {[f"{viz['x_column']} vs {viz['y_column']}" for viz in visualizations if 'x_column' in viz]}
        
        Please provide:
        1. Key insights about the relationships
        2. Strength and direction of associations
        3. Notable patterns or outliers
        4. Business implications of the relationships
        5. Recommendations for predictive modeling
        
        Focus on actionable insights from the relationship patterns.
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
                response_type="visualization",
                metadata={
                    "viz_type": "relationship",
                    "plot_count": len(visualizations)
                },
                artifacts={"visualizations": visualizations}
            )
        }
    
    async def _create_time_series_plots(self, 
                                      query: str, 
                                      context: SessionContext, 
                                      dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create time series visualizations"""
        
        # Find time columns
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
                    content="No time-based columns found for time series plots.",
                    response_type="info"
                )
            }
        
        time_col = time_columns[0]
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No numeric columns found for time series plots.",
                    response_type="info"
                )
            }
        
        # Ensure time column is datetime
        if dataset[time_col].dtype != 'datetime64[ns]':
            dataset = dataset.copy()
            dataset[time_col] = pd.to_datetime(dataset[time_col])
        
        # Sort by time
        dataset_sorted = dataset.sort_values(time_col)
        
        # Extract mentioned numeric columns
        mentioned_columns = [col for col in numeric_cols if col.lower() in query.lower()]
        if not mentioned_columns:
            mentioned_columns = numeric_cols[:3]  # Show first 3 if none specified
        
        visualizations = []
        
        # Individual time series plots
        for col in mentioned_columns:
            line_fig = px.line(
                dataset_sorted,
                x=time_col,
                y=col,
                title=f'{col} Over Time'
            )
            line_fig.update_layout(showlegend=False)
            
            visualizations.append({
                "type": "line",
                "time_column": time_col,
                "y_column": col,
                "title": f"{col} Over Time",
                "figure": line_fig.to_json()
            })
        
        # Combined time series plot if multiple columns
        if len(mentioned_columns) > 1:
            combined_fig = go.Figure()
            
            for col in mentioned_columns:
                combined_fig.add_trace(go.Scatter(
                    x=dataset_sorted[time_col],
                    y=dataset_sorted[col],
                    mode='lines',
                    name=col
                ))
            
            combined_fig.update_layout(
                title='Multiple Variables Over Time',
                xaxis_title=time_col,
                yaxis_title='Values'
            )
            
            visualizations.append({
                "type": "multi_line",
                "time_column": time_col,
                "y_columns": mentioned_columns,
                "title": "Multiple Variables Over Time",
                "figure": combined_fig.to_json()
            })
        
        # Generate insights
        insights_prompt = f"""
        Analyze these time series visualizations:
        
        Query: {query}
        Time column: {time_col}
        Variables plotted: {', '.join(mentioned_columns)}
        Time range: {dataset_sorted[time_col].min()} to {dataset_sorted[time_col].max()}
        
        Please provide:
        1. Key trends and patterns over time
        2. Seasonal or cyclical patterns if apparent
        3. Notable changes or anomalies
        4. Business implications of the temporal patterns
        5. Forecasting insights and recommendations
        
        Focus on actionable insights from the time series patterns.
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
                response_type="visualization",
                metadata={
                    "viz_type": "time_series",
                    "time_column": time_col,
                    "variables_plotted": mentioned_columns
                },
                artifacts={"visualizations": visualizations}
            )
        }
    
    async def _create_categorical_plots(self, 
                                      query: str, 
                                      context: SessionContext, 
                                      dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create categorical visualizations (bar charts, pie charts, etc.)"""
        
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="No categorical columns found for categorical plots.",
                    response_type="info"
                )
            }
        
        # Extract mentioned columns
        mentioned_columns = [col for col in categorical_cols if col.lower() in query.lower()]
        if not mentioned_columns:
            mentioned_columns = categorical_cols[:2]  # Show first 2 if none specified
        
        visualizations = []
        
        for col in mentioned_columns:
            value_counts = dataset[col].value_counts()
            
            # Limit to top categories if too many
            if len(value_counts) > 15:
                value_counts = value_counts.head(15)
                title_suffix = " (Top 15)"
            else:
                title_suffix = ""
            
            # Bar chart
            bar_fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Count of {col}{title_suffix}',
                labels={'x': col, 'y': 'Count'}
            )
            bar_fig.update_layout(showlegend=False)
            
            visualizations.append({
                "type": "bar",
                "column": col,
                "title": f"Count of {col}",
                "figure": bar_fig.to_json()
            })
            
            # Pie chart if reasonable number of categories
            if len(value_counts) <= 8:
                pie_fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f'Distribution of {col}'
                )
                
                visualizations.append({
                    "type": "pie",
                    "column": col,
                    "title": f"Distribution of {col}",
                    "figure": pie_fig.to_json()
                })
        
        # Cross-tabulation if multiple categorical columns
        if len(mentioned_columns) >= 2:
            col1, col2 = mentioned_columns[0], mentioned_columns[1]
            
            # Create cross-tabulation
            crosstab = pd.crosstab(dataset[col1], dataset[col2])
            
            # Heatmap
            heatmap_fig = px.imshow(
                crosstab.values,
                x=crosstab.columns,
                y=crosstab.index,
                title=f'{col1} vs {col2} Cross-tabulation',
                labels={'x': col2, 'y': col1, 'color': 'Count'},
                aspect="auto"
            )
            
            visualizations.append({
                "type": "heatmap",
                "x_column": col2,
                "y_column": col1,
                "title": f"{col1} vs {col2} Cross-tabulation",
                "figure": heatmap_fig.to_json()
            })
        
        # Generate insights
        insights_prompt = f"""
        Analyze these categorical visualizations:
        
        Query: {query}
        Columns visualized: {', '.join(mentioned_columns)}
        
        Category distributions:
        {[f"{col}: {dataset[col].nunique()} unique values" for col in mentioned_columns]}
        
        Please provide:
        1. Key insights about category distributions
        2. Dominant categories and their significance
        3. Patterns in category relationships if applicable
        4. Business implications of the distributions
        5. Recommendations for category-based strategies
        
        Focus on actionable insights from the categorical patterns.
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
                response_type="visualization",
                metadata={
                    "viz_type": "categorical",
                    "columns_plotted": mentioned_columns
                },
                artifacts={"visualizations": visualizations}
            )
        }
    
    async def _create_comparison_plots(self, 
                                     query: str, 
                                     context: SessionContext, 
                                     dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison visualizations"""
        
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Need both categorical and numeric columns for comparison plots.",
                    response_type="info"
                )
            }
        
        # Use first categorical as grouping variable
        group_col = categorical_cols[0]
        numeric_col = numeric_cols[0]
        
        # Extract mentioned columns if any
        mentioned_cats = [col for col in categorical_cols if col.lower() in query.lower()]
        mentioned_nums = [col for col in numeric_cols if col.lower() in query.lower()]
        
        if mentioned_cats:
            group_col = mentioned_cats[0]
        if mentioned_nums:
            numeric_col = mentioned_nums[0]
        
        visualizations = []
        
        # Box plot comparison
        box_fig = px.box(
            dataset,
            x=group_col,
            y=numeric_col,
            title=f'{numeric_col} by {group_col}'
        )
        
        visualizations.append({
            "type": "box_comparison",
            "x_column": group_col,
            "y_column": numeric_col,
            "title": f"{numeric_col} by {group_col}",
            "figure": box_fig.to_json()
        })
        
        # Bar chart of means
        means_by_group = dataset.groupby(group_col)[numeric_col].mean().reset_index()
        
        bar_means_fig = px.bar(
            means_by_group,
            x=group_col,
            y=numeric_col,
            title=f'Average {numeric_col} by {group_col}'
        )
        
        visualizations.append({
            "type": "bar_means",
            "x_column": group_col,
            "y_column": numeric_col,
            "title": f"Average {numeric_col} by {group_col}",
            "figure": bar_means_fig.to_json()
        })
        
        # Generate insights
        group_stats = dataset.groupby(group_col)[numeric_col].agg(['count', 'mean', 'std']).round(2)
        
        insights_prompt = f"""
        Analyze these comparison visualizations:
        
        Query: {query}
        Grouping variable: {group_col}
        Numeric variable: {numeric_col}
        
        Group statistics:
        {group_stats.to_dict()}
        
        Please provide:
        1. Key differences between groups
        2. Notable patterns in the comparisons
        3. Statistical significance of differences
        4. Business implications of group differences
        5. Recommendations based on group performance
        
        Focus on actionable insights from the group comparisons.
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
                response_type="visualization",
                metadata={
                    "viz_type": "comparison",
                    "group_column": group_col,
                    "numeric_column": numeric_col
                },
                artifacts={
                    "visualizations": visualizations,
                    "group_statistics": group_stats.to_dict()
                }
            )
        }
    
    async def _create_correlation_plots(self, 
                                      query: str, 
                                      context: SessionContext, 
                                      dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation visualizations"""
        
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "response": AgentResponse(
                    agent_name=self.name,
                    content="Need at least 2 numeric columns for correlation plots.",
                    response_type="info"
                )
            }
        
        # Calculate correlation matrix
        corr_matrix = dataset[numeric_cols].corr()
        
        visualizations = []
        
        # Correlation heatmap
        heatmap_fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            title='Correlation Matrix Heatmap',
            labels={'color': 'Correlation'},
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        heatmap_fig.update_layout(width=600, height=500)
        
        visualizations.append({
            "type": "correlation_heatmap",
            "title": "Correlation Matrix Heatmap",
            "figure": heatmap_fig.to_json()
        })
        
        # Generate insights
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        insights_prompt = f"""
        Analyze this correlation visualization:
        
        Query: {query}
        Variables analyzed: {len(numeric_cols)}
        
        Strong correlations (|r| > 0.5):
        {strong_correlations[:10]}  # Top 10
        
        Please provide:
        1. Key correlation patterns
        2. Strongest positive and negative correlations
        3. Potential multicollinearity concerns
        4. Business implications of correlations
        5. Recommendations for feature selection or modeling
        
        Focus on actionable insights from the correlation patterns.
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
                response_type="visualization",
                metadata={
                    "viz_type": "correlation",
                    "variables_analyzed": len(numeric_cols),
                    "strong_correlations": len(strong_correlations)
                },
                artifacts={
                    "visualizations": visualizations,
                    "correlation_matrix": corr_matrix.to_dict(),
                    "strong_correlations": strong_correlations
                }
            )
        }
    
    async def _create_dashboard(self, 
                              query: str, 
                              context: SessionContext, 
                              dataset: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive dashboard"""
        
        dashboard_prompt = f"""
        The user wants a dashboard overview. Based on this dataset:
        
        Shape: {dataset.shape}
        Numeric columns: {len(dataset.select_dtypes(include=[np.number]).columns)}
        Categorical columns: {len(dataset.select_dtypes(include=['object', 'category']).columns)}
        
        Query: {query}
        
        Please suggest:
        1. What types of visualizations would be most informative
        2. Key metrics to highlight
        3. Layout recommendations for the dashboard
        4. Priority order of visualizations
        5. Interactive elements that would be valuable
        
        Provide a strategic approach to dashboard creation.
        """
        
        response_text = await self.generate_response(
            dashboard_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="guidance",
                metadata={"viz_type": "dashboard"}
            )
        }
    
    async def _suggest_visualization(self, 
                                   query: str, 
                                   context: SessionContext, 
                                   dataset: pd.DataFrame) -> Dict[str, Any]:
        """Suggest appropriate visualizations"""
        
        data_summary = {
            "shape": dataset.shape,
            "numeric_columns": list(dataset.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": [col for col in dataset.columns 
                               if dataset[col].dtype in ['datetime64[ns]', 'datetime64']]
        }
        
        suggestion_prompt = f"""
        Suggest appropriate visualizations for this request:
        
        Query: {query}
        
        Dataset characteristics:
        {data_summary}
        
        Please provide:
        1. Most suitable visualization types for this query
        2. Specific column recommendations
        3. Alternative visualization options
        4. Expected insights from each suggested visualization
        5. Best practices for the recommended visualizations
        
        Be specific and actionable in your recommendations.
        """
        
        response_text = await self.generate_response(
            suggestion_prompt,
            context.get_recent_context()
        )
        
        return {
            "success": True,
            "response": AgentResponse(
                agent_name=self.name,
                content=response_text,
                response_type="suggestion",
                metadata={"viz_type": "suggestion"}
            )
        }
    
    def _no_data_response(self) -> Dict[str, Any]:
        """Response when no data is available"""
        return {
            "success": False,
            "response": AgentResponse(
                agent_name=self.name,
                content="No dataset is currently loaded. Please load a dataset first to create visualizations.",
                response_type="info"
            )
        }
