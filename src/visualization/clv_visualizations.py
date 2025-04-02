# In clv_visualizations.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_clv_distribution_chart(customer_data):
    """
    Create histogram of customer lifetime values
    
    Args:
        customer_data (DataFrame): DataFrame with CLV values
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.histogram(
        customer_data,
        x='clv_1yr',
        nbins=50,
        title='Customer Lifetime Value Distribution',
        labels={'clv_1yr': 'Customer Lifetime Value (1 Year)'},
        color_discrete_sequence=['darkblue']
    )
    
    fig.update_layout(
        xaxis_title='CLV (Currency Units)',
        yaxis_title='Number of Customers',
        bargap=0.1
    )
    
    return fig