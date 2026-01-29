import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from leakage_detector import LeakageDetector

def load_columns(file):
    try:
        df = pd.read_csv(file.name, nrows=5) # Read only header/small sample
        cols = list(df.columns)
        return gr.update(choices=cols, value=cols[-1] if cols else None), \
               gr.update(choices=["None"] + cols), \
               gr.update(choices=["None"] + cols)
    except Exception as e:
        return gr.update(), gr.update(), gr.update()

def analyze_leakage(file, target_col, time_col, id_col):
    if file is None or not target_col:
        return "Please upload a CSV and select a target column.", None, None, None, None

    try:
        # Load data (cap at 100k rows for performance in Spaces)
        df = pd.read_csv(file.name, nrows=100000)
        
        # Handle optional columns
        t_col = time_col if time_col != "None" else None
        i_col = id_col if id_col != "None" else None
        
        detector = LeakageDetector(df, target_col=target_col, time_col=t_col, id_col=i_col)
        report = detector.run_all()
        
        # 1. Summary
        summary_text = f"## Overall Risk Score: {report['overall_score']}/100\n"
        summary_text += f"### Severity: {report['severity']}\n\n"
        if report['summary']:
            summary_text += "**Warnings:**\n" + "\n".join([f"- {s}" for s in report['summary']])

        # 2. DataFrame for Table
        # Convert nested feature dict to dataframe
        rows = []
        for feature, details in report['features'].items():
            rows.append({
                "Feature": feature,
                "Risk Level": details['risk'],
                "Correlation": round(details['corr'], 3),
                "Mutual Info": round(details['mi'], 3),
                "Predictive Strength": round(details['tree_score'], 3),
                "Reasons": "; ".join(details['reasons'])
            })
        
        res_df = pd.DataFrame(rows)
        # Sort by Risk (Critical first)
        risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        res_df['sort_key'] = res_df['Risk Level'].map(risk_order)
        res_df = res_df.sort_values('sort_key').drop('sort_key', axis=1)

        # 3. Correlation Heatmap
        fig_corr = plt.figure(figsize=(10, 8))
        # Recalculate corr for display (subset of top risky features + target)
        top_features = res_df.head(10)['Feature'].tolist()
        if target_col not in top_features:
            top_features.append(target_col)
            
        # We need to re-encode for this vis
        disp_df = df[top_features].copy()
        for c in disp_df.select_dtypes(include=['object']):
            disp_df[c] = pd.factorize(disp_df[c])[0]
            
        sns.heatmap(disp_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap (Top Risky Features)")
        plt.tight_layout()

        # 4. Feature Risk Bar Chart
        fig_bar = plt.figure(figsize=(10, 6))
        sns.barplot(data=res_df.head(15), x='Predictive Strength', y='Feature', hue='Risk Level', dodge=False, palette='viridis')
        plt.title("Top Features by Predictive Strength")
        plt.tight_layout()

        return summary_text, res_df, fig_corr, fig_bar, "Analysis Complete"

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, "Failed"

# UI Layout
with gr.Blocks(title="Data Leakage Early Warning System") as app:
    gr.Markdown("# 🛡️ Data Leakage Early Warning System")
    gr.Markdown("Upload your dataset to detect Target Leakage, Time Leakage, and Proxy features before training.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
            target_dropdown = gr.Dropdown(label="Select Target Column", interactive=True)
            time_dropdown = gr.Dropdown(label="Select Time Column (Optional)", interactive=True)
            id_dropdown = gr.Dropdown(label="Select Entity ID Column (Optional)", interactive=True)
            analyze_btn = gr.Button("🔍 Analyze Leakage", variant="primary")
        
        with gr.Column(scale=2):
            status_output = gr.Markdown("Waiting for input...")
            summary_output = gr.Markdown("")
    
    with gr.Tabs():
        with gr.TabItem("Detailed Report"):
            result_table = gr.Dataframe(label="Feature Risk Analysis")
        with gr.TabItem("Visualizations"):
            with gr.Row():
                corr_plot = gr.Plot(label="Correlation Heatmap")
                risk_plot = gr.Plot(label="Feature Risk Chart")

    # Interactivity
    file_input.change(load_columns, inputs=file_input, outputs=[target_dropdown, time_dropdown, id_dropdown])
    analyze_btn.click(analyze_leakage, 
                      inputs=[file_input, target_dropdown, time_dropdown, id_dropdown], 
                      outputs=[summary_output, result_table, corr_plot, risk_plot, status_output])

if __name__ == "__main__":
    app.launch()
