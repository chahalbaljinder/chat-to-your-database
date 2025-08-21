"""
Data loading and processing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
from config.settings import Config
from utils.session_utils import DatasetInfo

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic processing of various data formats"""
    
    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load data file and create dataset info"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)")
        
        # Load based on file extension
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif extension == '.json':
                df = pd.read_json(file_path)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            # Check row limit
            if len(df) > Config.MAX_ROWS:
                logger.warning(f"Dataset has {len(df)} rows, truncating to {Config.MAX_ROWS}")
                df = df.head(Config.MAX_ROWS)
            
            # Create dataset info
            dataset_info = DataLoader._create_dataset_info(df, file_path)
            
            logger.info(f"Loaded dataset: {dataset_info.name} with shape {dataset_info.shape}")
            return df, dataset_info
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def _create_dataset_info(df: pd.DataFrame, file_path: Path) -> DatasetInfo:
        """Create comprehensive dataset information"""
        from utils.session_utils import convert_numpy_types
        
        # Basic info
        dataset_info = DatasetInfo(
            name=file_path.stem,
            file_path=str(file_path),
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()}
        )
        
        # Generate summary statistics
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_summary = df[numeric_cols].describe().to_dict()
                missing_values = df.isnull().sum().to_dict()
                unique_values = df.nunique().to_dict()
                
                # Convert numpy types to native Python types
                dataset_info.summary_stats = convert_numpy_types({
                    "numeric_summary": numeric_summary,
                    "missing_values": missing_values,
                    "unique_values": unique_values
                })
            else:
                missing_values = df.isnull().sum().to_dict()
                unique_values = df.nunique().to_dict()
                
                # Convert numpy types to native Python types
                dataset_info.summary_stats = convert_numpy_types({
                    "missing_values": missing_values,
                    "unique_values": unique_values
                })
        except Exception as e:
            logger.warning(f"Could not generate summary statistics: {str(e)}")
            dataset_info.summary_stats = {"error": str(e)}
        
        # Data quality assessment
        dataset_info.quality_info = convert_numpy_types(DataLoader._assess_data_quality(df))
        
        return dataset_info
    
    @staticmethod
    def _assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and identify issues"""
        from utils.session_utils import convert_numpy_types
        
        quality_info = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "issues": []
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.5]
        if len(high_missing) > 0:
            quality_info["issues"].append({
                "type": "high_missing_values",
                "columns": convert_numpy_types(high_missing.to_dict()),
                "description": "Columns with >50% missing values"
            })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_info["issues"].append({
                "type": "duplicate_rows",
                "count": int(duplicate_count),
                "description": f"{duplicate_count} duplicate rows found"
            })
        
        # Check for potential ID columns
        potential_ids = []
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].dtype in ['int64', 'object']:
                potential_ids.append(col)
        
        if potential_ids:
            quality_info["potential_id_columns"] = potential_ids
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_info["issues"].append({
                "type": "constant_columns",
                "columns": constant_cols,
                "description": "Columns with constant or single unique value"
            })
        
        # Check data types
        text_as_numeric = []
        for col in df.select_dtypes(include=['object']):
            try:
                pd.to_numeric(df[col], errors='raise')
                text_as_numeric.append(col)
            except:
                pass
        
        if text_as_numeric:
            quality_info["suggestions"] = quality_info.get("suggestions", [])
            quality_info["suggestions"].append({
                "type": "type_conversion",
                "columns": text_as_numeric,
                "description": "Text columns that could be converted to numeric"
            })
        
        return quality_info

class DataProfiler:
    """Advanced data profiling and analysis"""
    
    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data profiling"""
        profile = {
            "overview": DataProfiler._get_overview(df),
            "columns": DataProfiler._profile_columns(df),
            "correlations": DataProfiler._get_correlations(df),
            "patterns": DataProfiler._detect_patterns(df)
        }
        
        return profile
    
    @staticmethod
    def _get_overview(df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset overview"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "completeness": (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    
    @staticmethod
    def _profile_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Profile individual columns"""
        column_profiles = {}
        
        for col in df.columns:
            series = df[col]
            profile = {
                "dtype": str(series.dtype),
                "non_null_count": series.count(),
                "null_count": series.isnull().sum(),
                "unique_count": series.nunique(),
                "completeness": (series.count() / len(df)) * 100
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(series):
                profile.update({
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "outliers": DataProfiler._detect_outliers(series)
                })
            
            # Categorical columns
            elif pd.api.types.is_object_dtype(series):
                value_counts = series.value_counts().head(10)
                profile.update({
                    "top_values": value_counts.to_dict(),
                    "avg_length": series.str.len().mean() if series.dtype == 'object' else None
                })
            
            column_profiles[col] = profile
        
        return column_profiles
    
    @staticmethod
    def _get_correlations(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_val
                        })
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": high_correlations
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect common patterns in the data"""
        patterns = {}
        
        # Time series detection
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            patterns["time_series"] = {
                "columns": datetime_cols.tolist(),
                "date_range": {
                    col: {
                        "start": df[col].min(),
                        "end": df[col].max(),
                        "span_days": (df[col].max() - df[col].min()).days
                    }
                    for col in datetime_cols
                }
            }
        
        # Hierarchical data detection
        potential_hierarchies = []
        for col in df.select_dtypes(include=['object']).columns:
            if '/' in df[col].astype(str).iloc[0] or '.' in df[col].astype(str).iloc[0]:
                potential_hierarchies.append(col)
        
        if potential_hierarchies:
            patterns["hierarchical_columns"] = potential_hierarchies
        
        return patterns
    
    @staticmethod
    def _detect_outliers(series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            return {
                "count": len(outliers),
                "percentage": (len(outliers) / len(series)) * 100,
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        except:
            return {"count": 0, "percentage": 0.0}
