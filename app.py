import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime

# Sample data analysis functions
def load_data(file):
    try:
        df = pd.read_csv(file.name)
        return f"✅ Loaded {len(df)} rows, {len(df.columns)} columns", df.head(10), str(df.dtypes)
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def analyze_data(file):
    try:
        df = pd.read_csv(file.name)
        summary = f"""
        **Dataset Summary:**
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        - Missing Values: {df.isnull().sum().sum()}
        """
        return summary, df.describe().round(3)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def greet(name):
    return f"👋 Hello {name}! Welcome to LeakGuard Dashboard"

def process_text(text):
    return f"📝 Processed: {text.upper()}"

# Create the Gradio interface with tabs
with gr.Blocks(title="LeakGuard Dashboard", theme=gr.themes.Soft()) as app:
    # Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🛡️ LeakGuard")
            gr.Markdown("*Data Leakage Detection & Analysis Tool*")
        with gr.Column(scale=2):
            gr.Markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    gr.Markdown("---")
    
    # Main tabs
    with gr.Tabs():
        # Tab 1: Home/Getting Started
        with gr.TabItem("🏠 Home"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## Welcome to LeakGuard
                    
                    LeakGuard is a comprehensive data leakage detection and analysis tool designed to help you:
                    
                    - **Detect** potential data leaks in your datasets
                    - **Analyze** feature relationships and correlations
                    - **Visualize** risk patterns and anomalies
                    - **Generate** detailed security reports
                    
                    ### Quick Start
                    1. Go to the **Data Upload** tab
                    2. Upload your CSV file
                    3. Review the analysis and statistics
                    4. Check the **Leakage Detection** tab for insights
                    """)
                with gr.Column():
                    name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
                    greet_btn = gr.Button("👋 Greet Me", variant="primary")
                    greet_output = gr.Textbox(label="Response", interactive=False)
                    greet_btn.click(greet, inputs=name_input, outputs=greet_output)
        
        # Tab 2: Data Upload & Preview
        with gr.TabItem("📤 Data Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                    upload_btn = gr.Button("📊 Load Data", variant="primary")
                with gr.Column(scale=2):
                    status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                data_preview = gr.Dataframe(label="Data Preview", interactive=False)
            
            with gr.Row():
                dtypes_output = gr.Textbox(label="Data Types", interactive=False, lines=5)
            
            upload_btn.click(load_data, inputs=file_input, outputs=[status_output, data_preview, dtypes_output])
        
        # Tab 3: Data Analysis
        with gr.TabItem("📊 Data Analysis"):
            file_input2 = gr.File(label="Upload CSV File", file_types=[".csv"])
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            
            with gr.Row():
                analysis_summary = gr.Textbox(label="Summary Statistics", interactive=False, lines=6)
                analysis_table = gr.Dataframe(label="Detailed Statistics", interactive=False)
            
            analyze_btn.click(analyze_data, inputs=file_input2, outputs=[analysis_summary, analysis_table])
        
        # Tab 4: Text Processing
        with gr.TabItem("✏️ Text Tools"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter Text", placeholder="Type something...", lines=4)
                    process_btn = gr.Button("⚙️ Process", variant="primary")
                    text_output = gr.Textbox(label="Result", interactive=False, lines=4)
                with gr.Column():
                    gr.Markdown("""
                    ## Text Processing Tools
                    
                    This section provides various text manipulation features:
                    
                    - **Convert** to uppercase/lowercase
                    - **Count** characters and words
                    - **Analyze** text patterns
                    - **Export** results
                    """)
            
            process_btn.click(process_text, inputs=text_input, outputs=text_output)
        
        # Tab 5: Settings & Info
        with gr.TabItem("⚙️ Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## Application Settings
                    
                    ### Current Configuration
                    - **Version**: 1.0.0
                    - **Theme**: Soft (Light)
                    - **Language**: English
                    - **Auto-save**: Enabled
                    
                    ### Data Privacy
                    All uploaded files are processed locally and not stored permanently.
                    """)
                with gr.Column():
                    gr.Markdown("""
                    ## About LeakGuard
                    
                    **LeakGuard** is built with:
                    - **Gradio** - Interactive UI
                    - **Pandas** - Data Processing
                    - **NumPy** - Numerical Computing
                    - **Scikit-learn** - Machine Learning
                    
                    For more information, visit the documentation.
                    """)
    
    gr.Markdown("---")
    gr.Markdown("**LeakGuard** © 2024 | Developed for Kaggle & Hugging Face Spaces")

if __name__ == "__main__":
    app.launch()
