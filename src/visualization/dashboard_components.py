"""
Base visualization components for BFSI dashboards.
Contains reusable functions for creating consistent visualization elements.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent theme and colors
BANK_COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'tertiary': '#2ca02c',     # Green
    'quaternary': '#d62728',   # Red
    'quinary': '#9467bd',      # Purple
    'background': '#f7f7f7',   # Light gray
    'text': '#333333'          # Dark gray
}

SEGMENT_COLORS = {
    0: '#1f77b4',  # Loyal Savers
    1: '#ff7f0e',  # Active Users
    2: '#2ca02c',  # Transactors
    3: '#d62728',  # Credit Utilizers
    4: '#9467bd'   # Low Engagement
}

def apply_bfsi_theme(fig):
    """
    Apply consistent BFSI-themed styling to a Plotly figure
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to style
        
    Returns:
        plotly.graph_objects.Figure: Styled figure
    """
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12, color=BANK_COLORS['text']),
        paper_bgcolor=BANK_COLORS['background'],
        plot_bgcolor=BANK_COLORS['background'],
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bordercolor=BANK_COLORS['text'],
            borderwidth=1
        ),
        colorway=[BANK_COLORS['primary'], BANK_COLORS['secondary'], 
                 BANK_COLORS['tertiary'], BANK_COLORS['quaternary'], 
                 BANK_COLORS['quinary']]
    )
    
    fig.update_xaxes(
        gridcolor='lightgray',
        zerolinecolor='lightgray'
    )
    
    fig.update_yaxes(
        gridcolor='lightgray',
        zerolinecolor='lightgray'
    )
    
    return fig

def create_metric_card(title, value, subtitle=None, color=BANK_COLORS['primary']):
    """
    Create a single metric card for dashboards
    
    Args:
        title (str): Metric title
        value (str/float/int): Metric value to display
        subtitle (str, optional): Subtitle or context for the metric
        color (str): Color for the metric value
        
    Returns:
        plotly.graph_objects.Figure: Metric card as a Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=value,
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{subtitle if subtitle else ''}</span>"},
        number={"font": {"size": 50, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=BANK_COLORS['background']
    )
    
    return fig

def create_segmented_bar(data, x_col, y_col, segment_col, title, xaxis_title=None, yaxis_title=None):
    """
    Create a bar chart segmented by customer segment
    
    Args:
        data (pd.DataFrame): Data containing the columns to plot
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis values
        segment_col (str): Column name for segmentation (usually 'cluster')
        title (str): Chart title
        xaxis_title (str, optional): Custom x-axis title
        yaxis_title (str, optional): Custom y-axis title
        
    Returns:
        plotly.graph_objects.Figure: Segmented bar chart
    """
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col,
        color=segment_col,
        title=title,
        color_discrete_map=SEGMENT_COLORS,
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title=xaxis_title if xaxis_title else x_col,
        yaxis_title=yaxis_title if yaxis_title else y_col
    )
    
    return apply_bfsi_theme(fig)

def format_currency(value):
    """
    Format a number as currency
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${value:,.2f}"

def format_percent(value):
    """
    Format a number as percentage
    
    Args:
        value (float): Value to format (0-1)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value*100:.1f}%"