# src/reporting/generate_segment_report.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_segment_report():
    """Generate comprehensive segmentation report with visualizations."""
    # Load segmentation results
    segments = pd.read_csv('../data/processed/customer_segments.csv')
    profiles = pd.read_csv('../data/processed/cluster_profiles.csv')
    
    # Create output directory
    os.makedirs('../reports/figures', exist_ok=True)
    
    # Generate segment distribution chart
    plt.figure(figsize=(10, 6))
    segment_counts = segments['segment_name'].value_counts()
    ax = segment_counts.plot(kind='bar', color='navy')
    plt.title('Customer Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentages on top of bars
    total = len(segments)
    for i, count in enumerate(segment_counts):
        percentage = count / total * 100
        ax.text(i, count + 10, f"{percentage:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/segment_distribution.png', dpi=300)
    
    # Generate additional visualizations...
    
    # Create HTML report
    with open('../reports/segment_report.html', 'w') as f:
        f.write('''
        <html>
        <head>
            <title>Banking Customer Segmentation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #00356B; }
                .metric { margin: 20px 0; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Banking Customer Segmentation Report</h1>
            <p>Generated on: {date}</p>
            
            <h2>Segment Distribution</h2>
            <img src="figures/segment_distribution.png" alt="Segment Distribution">
            
            <!-- Additional sections here -->
            
        </body>
        </html>
        '''.format(date=pd.Timestamp.now().strftime('%Y-%m-%d')))
    
    print("Segment report generated successfully.")

if __name__ == "__main__":
    generate_segment_report()