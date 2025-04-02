# In segmentation_visualizations.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_segment_distribution_chart(segment_data):
    """
    Create pie chart showing customer segment distribution
    
    Args:
        segment_data (DataFrame): DataFrame with customer segments
        
    Returns:
        plotly.graph_objects.Figure
    """
    segment_counts = segment_data['cluster'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    fig = px.pie(
        segment_counts, 
        values='Count', 
        names='Segment',
        title='Customer Segment Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        legend_title="Segment",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    
    return fig

def create_segment_comparison_chart(segment_profiles):
    """
    Create radar chart comparing key metrics across segments
    
    Args:
        segment_profiles (DataFrame): DataFrame with segment profiles
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Implementation here