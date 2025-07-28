"""
Advanced Data Visualization Engine
Interactive and intelligent data visualization with automatic insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = "seaborn"
    color_palette: str = "viridis"
    interactive: bool = False
    save_format: str = "png"

class DataAnalyzer:
    """Analyzes data to suggest appropriate visualizations."""
    
    def __init__(self):
        self.column_types = {}
        self.data_insights = {}
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataframe structure and content."""
        insights = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Statistical insights for numeric columns
        if insights["numeric_columns"]:
            insights["numeric_stats"] = df[insights["numeric_columns"]].describe().to_dict()
        
        # Categorical insights
        categorical_insights = {}
        for col in insights["categorical_columns"]:
            unique_count = df[col].nunique()
            categorical_insights[col] = {
                "unique_count": unique_count,
                "top_values": df[col].value_counts().head().to_dict(),
                "is_high_cardinality": unique_count > 50
            }
        insights["categorical_insights"] = categorical_insights
        
        return insights
    
    def suggest_visualizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data."""
        insights = self.analyze_dataframe(df)
        suggestions = []
        
        # Single variable suggestions
        for col in insights["numeric_columns"]:
            suggestions.append({
                "type": "histogram",
                "columns": [col],
                "title": f"Distribution of {col}",
                "description": f"Shows the distribution of values in {col}"
            })
            
            suggestions.append({
                "type": "boxplot",
                "columns": [col],
                "title": f"Boxplot of {col}",
                "description": f"Shows outliers and quartiles for {col}"
            })
        
        for col in insights["categorical_columns"]:
            if not insights["categorical_insights"][col]["is_high_cardinality"]:
                suggestions.append({
                    "type": "countplot",
                    "columns": [col],
                    "title": f"Count of {col}",
                    "description": f"Shows frequency of categories in {col}"
                })
        
        # Two variable suggestions
        numeric_cols = insights["numeric_columns"]
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    suggestions.append({
                        "type": "scatter",
                        "columns": [numeric_cols[i], numeric_cols[j]],
                        "title": f"{numeric_cols[i]} vs {numeric_cols[j]}",
                        "description": f"Relationship between {numeric_cols[i]} and {numeric_cols[j]}"
                    })
        
        # Correlation heatmap for multiple numeric variables
        if len(numeric_cols) > 2:
            suggestions.append({
                "type": "correlation_heatmap",
                "columns": numeric_cols,
                "title": "Correlation Matrix",
                "description": "Shows correlations between all numeric variables"
            })
        
        return suggestions

class BaseVisualizer(ABC):
    """Abstract base class for visualizers."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        plt.style.use(config.style)
    
    @abstractmethod
    def create_visualization(self, data: Any, **kwargs) -> plt.Figure:
        """Create the visualization."""
        pass
    
    def setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Set up figure and axes."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        return fig, ax

class HistogramVisualizer(BaseVisualizer):
    """Creates histogram visualizations."""
    
    def create_visualization(self, data: pd.Series, **kwargs) -> plt.Figure:
        fig, ax = self.setup_figure()
        
        ax.hist(data.dropna(), bins=kwargs.get('bins', 30), 
                color=kwargs.get('color', 'skyblue'), alpha=0.7)
        ax.set_xlabel(data.name)
        ax.set_ylabel('Frequency')
        ax.set_title(kwargs.get('title', f'Distribution of {data.name}'))
        ax.grid(True, alpha=0.3)
        
        return fig

class ScatterVisualizer(BaseVisualizer):
    """Creates scatter plot visualizations."""
    
    def create_visualization(self, x_data: pd.Series, y_data: pd.Series, **kwargs) -> plt.Figure:
        fig, ax = self.setup_figure()
        
        scatter = ax.scatter(x_data, y_data, 
                           c=kwargs.get('color_col', None),
                           cmap=self.config.color_palette,
                           alpha=0.6)
        
        ax.set_xlabel(x_data.name)
        ax.set_ylabel(y_data.name)
        ax.set_title(kwargs.get('title', f'{x_data.name} vs {y_data.name}'))
        
        if kwargs.get('color_col') is not None:
            plt.colorbar(scatter)
        
        return fig

class CorrelationHeatmapVisualizer(BaseVisualizer):
    """Creates correlation heatmap visualizations."""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        fig, ax = self.setup_figure()
        
        correlation_matrix = data.corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap=self.config.color_palette,
                   center=0,
                   ax=ax)
        
        ax.set_title(kwargs.get('title', 'Correlation Matrix'))
        
        return fig

class BoxplotVisualizer(BaseVisualizer):
    """Creates boxplot visualizations."""
    
    def create_visualization(self, data: pd.Series, **kwargs) -> plt.Figure:
        fig, ax = self.setup_figure()
        
        ax.boxplot(data.dropna())
        ax.set_ylabel(data.name)
        ax.set_title(kwargs.get('title', f'Boxplot of {data.name}'))
        ax.grid(True, alpha=0.3)
        
        return fig

class TimeSeriesVisualizer(BaseVisualizer):
    """Creates time series visualizations."""
    
    def create_visualization(self, data: pd.Series, **kwargs) -> plt.Figure:
        fig, ax = self.setup_figure()
        
        ax.plot(data.index, data.values, linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel(data.name)
        ax.set_title(kwargs.get('title', f'Time Series of {data.name}'))
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if isinstance(data.index, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        return fig

class InteractiveVisualizer:
    """Creates interactive visualizations using Plotly."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            self.go = go
            self.px = px
            self.available = True
        except ImportError:
            self.available = False
    
    def create_interactive_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs):
        """Create interactive scatter plot."""
        if not self.available:
            raise ImportError("Plotly not available for interactive visualizations")
        
        fig = self.px.scatter(df, x=x_col, y=y_col,
                             color=kwargs.get('color_col'),
                             size=kwargs.get('size_col'),
                             hover_data=kwargs.get('hover_data'),
                             title=kwargs.get('title', f'{x_col} vs {y_col}'))
        
        return fig
    
    def create_interactive_line(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs):
        """Create interactive line plot."""
        if not self.available:
            raise ImportError("Plotly not available for interactive visualizations")
        
        fig = self.px.line(df, x=x_col, y=y_col,
                          color=kwargs.get('color_col'),
                          title=kwargs.get('title', f'{y_col} over {x_col}'))
        
        return fig

class VisualizationEngine:
    """Main visualization engine that orchestrates all components."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.analyzer = DataAnalyzer()
        self.interactive_viz = InteractiveVisualizer(self.config)
        
        # Initialize visualizers
        self.visualizers = {
            "histogram": HistogramVisualizer(self.config),
            "scatter": ScatterVisualizer(self.config),
            "correlation_heatmap": CorrelationHeatmapVisualizer(self.config),
            "boxplot": BoxplotVisualizer(self.config),
            "timeseries": TimeSeriesVisualizer(self.config)
        }
    
    def auto_visualize(self, df: pd.DataFrame, max_plots: int = 5) -> List[plt.Figure]:
        """Automatically create visualizations based on data analysis."""
        suggestions = self.analyzer.suggest_visualizations(df)
        figures = []
        
        for i, suggestion in enumerate(suggestions[:max_plots]):
            try:
                fig = self.create_visualization(df, suggestion)
                if fig:
                    figures.append(fig)
            except Exception as e:
                print(f"Error creating visualization {suggestion['type']}: {e}")
        
        return figures
    
    def create_visualization(self, df: pd.DataFrame, suggestion: Dict[str, Any]) -> Optional[plt.Figure]:
        """Create a specific visualization based on suggestion."""
        viz_type = suggestion["type"]
        columns = suggestion["columns"]
        
        if viz_type == "histogram":
            return self.visualizers["histogram"].create_visualization(
                df[columns[0]], title=suggestion["title"]
            )
        
        elif viz_type == "scatter":
            return self.visualizers["scatter"].create_visualization(
                df[columns[0]], df[columns[1]], title=suggestion["title"]
            )
        
        elif viz_type == "correlation_heatmap":
            return self.visualizers["correlation_heatmap"].create_visualization(
                df[columns], title=suggestion["title"]
            )
        
        elif viz_type == "boxplot":
            return self.visualizers["boxplot"].create_visualization(
                df[columns[0]], title=suggestion["title"]
            )
        
        elif viz_type == "timeseries":
            return self.visualizers["timeseries"].create_visualization(
                df[columns[0]], title=suggestion["title"]
            )
        
        return None
    
    def create_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive dashboard of visualizations."""
        insights = self.analyzer.analyze_dataframe(df)
        suggestions = self.analyzer.suggest_visualizations(df)
        
        dashboard = {
            "data_summary": insights,
            "visualizations": [],
            "insights": []
        }
        
        # Create visualizations
        figures = self.auto_visualize(df, max_plots=8)
        dashboard["visualizations"] = figures
        
        # Generate automatic insights
        dashboard["insights"] = self._generate_insights(df, insights)
        
        return dashboard
    
    def _generate_insights(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Generate automatic insights from data analysis."""
        insights = []
        
        # Data quality insights
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 10:
            insights.append(f"Data has {missing_pct:.1f}% missing values - consider data cleaning")
        
        # Numeric insights
        for col in analysis["numeric_columns"]:
            stats = analysis["numeric_stats"][col]
            if stats["std"] / stats["mean"] > 2:
                insights.append(f"{col} has high variability (CV > 2)")
            
            # Check for potential outliers
            q1, q3 = stats["25%"], stats["75%"]
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            if stats["max"] > outlier_threshold:
                insights.append(f"{col} may have outliers (max value significantly above Q3)")
        
        # Categorical insights
        for col, info in analysis["categorical_insights"].items():
            if info["is_high_cardinality"]:
                insights.append(f"{col} has high cardinality ({info['unique_count']} unique values)")
        
        return insights
    
    def save_visualizations(self, figures: List[plt.Figure], prefix: str = "viz") -> List[str]:
        """Save visualizations to files."""
        saved_files = []
        
        for i, fig in enumerate(figures):
            filename = f"{prefix}_{i+1}.{self.config.save_format}"
            fig.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)  # Free memory
        
        return saved_files

# Example usage
if __name__ == "__main__":
    print("Testing Advanced Data Visualization Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Initialize visualization engine
    config = VisualizationConfig(
        figure_size=(12, 8),
        style="seaborn",
        color_palette="viridis"
    )
    
    viz_engine = VisualizationEngine(config)
    
    # Analyze data
    print("Analyzing data...")
    insights = viz_engine.analyzer.analyze_dataframe(df)
    print(f"Data shape: {insights['shape']}")
    print(f"Numeric columns: {insights['numeric_columns']}")
    print(f"Categorical columns: {insights['categorical_columns']}")
    
    # Get visualization suggestions
    suggestions = viz_engine.analyzer.suggest_visualizations(df)
    print(f"\nGenerated {len(suggestions)} visualization suggestions")
    
    # Create automatic visualizations
    print("Creating automatic visualizations...")
    figures = viz_engine.auto_visualize(df, max_plots=5)
    print(f"Created {len(figures)} visualizations")
    
    # Create full dashboard
    dashboard = viz_engine.create_dashboard(df)
    print(f"\nDashboard insights:")
    for insight in dashboard["insights"]:
        print(f"- {insight}")
    
    # Clean up
    for fig in figures:
        plt.close(fig)
    
    print("\nAdvanced Data Visualization Engine implemented successfully! 🚀")
