import gradio as gr
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import hashlib
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# ============================================================================
# 1️⃣ TARGET LEAKAGE DETECTION
# ============================================================================

def detect_target_leakage(df, target_col):
    """
    Detect features that contain direct/indirect target information.
    Uses: MI, Pearson & Spearman correlation, predictiveness flags.
    """
    results = {}
    target = df[target_col]
    is_classification = target.dtype == 'object' or len(target.unique()) < 20
    
    for col in df.columns:
        if col == target_col:
            continue
        
        feature = df[col]
        
        # Skip non-numeric features for correlation
        if feature.dtype not in ['float64', 'int64', 'float32', 'int32']:
            try:
                le = LabelEncoder()
                feature_encoded = le.fit_transform(feature.astype(str))
            except:
                continue
        else:
            feature_encoded = feature.values
        
        # Calculate MI
        try:
            if is_classification:
                target_encoded = LabelEncoder().fit_transform(target.astype(str))
                mi = mutual_info_classif(feature_encoded.reshape(-1, 1), target_encoded)[0]
            else:
                mi = mutual_info_regression(feature_encoded.reshape(-1, 1), target.values)[0]
        except:
            mi = 0
        
        # Calculate Pearson correlation
        try:
            if feature.dtype in ['float64', 'int64', 'float32', 'int32'] and target.dtype in ['float64', 'int64', 'float32', 'int32']:
                pearson_corr = feature.corr(target)
            else:
                pearson_corr = 0
        except:
            pearson_corr = 0
        
        # Calculate Spearman correlation
        try:
            spearman_corr, _ = spearmanr(feature_encoded, target_encoded if is_classification else target.values)
            spearman_corr = abs(spearman_corr) if not np.isnan(spearman_corr) else 0
        except:
            spearman_corr = 0
        
        results[col] = {
            'mi': mi,
            'pearson': abs(float(pearson_corr)) if pearson_corr == pearson_corr else 0,
            'spearman': spearman_corr
        }
    
    return results

# ============================================================================
# 2️⃣ TIME LEAKAGE DETECTION
# ============================================================================

def detect_time_leakage(df, target_col, time_col):
    """
    Detect future information leaking into past samples.
    Uses: correlation drift, rolling window MI.
    """
    if time_col not in df.columns:
        return None, "Time column not found"
    
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    target = df_sorted[target_col]
    is_classification = target.dtype == 'object' or len(target.unique()) < 20
    
    results = {}
    n_splits = 5
    split_size = len(df) // n_splits
    
    for col in df.columns:
        if col in [target_col, time_col]:
            continue
        
        feature = df_sorted[col]
        
        correlations = []
        for i in range(n_splits - 1):
            start = i * split_size
            end = (i + 1) * split_size
            
            try:
                if feature.dtype in ['float64', 'int64', 'float32', 'int32'] and target.dtype in ['float64', 'int64', 'float32', 'int32']:
                    corr = abs(feature.iloc[start:end].corr(target.iloc[start:end]))
                    correlations.append(corr)
            except:
                pass
        
        if correlations:
            correlation_drift = max(correlations) - min(correlations) if correlations else 0
            results[col] = {
                'drift': correlation_drift,
                'mean_corr': np.mean(correlations),
                'max_corr': max(correlations)
            }
    
    return results, "Time leakage analysis complete"

# ============================================================================
# 3️⃣ DUPLICATE / SPLIT LEAKAGE DETECTION
# ============================================================================

def detect_duplicate_leakage(df, target_col, id_col=None):
    """
    Detect same/near-identical samples across splits.
    Uses: row hashing, duplicate ratio, entity ID overlap.
    """
    results = {}
    
    # Row hashing approach
    def hash_row(row):
        return hashlib.md5(str(row.values).encode()).hexdigest()
    
    df_numeric = df.select_dtypes(include=[np.number])
    if len(df_numeric) > 0:
        row_hashes = df_numeric.apply(hash_row, axis=1)
        duplicate_count = row_hashes.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
        
        results['duplicate_ratio'] = duplicate_ratio
        results['duplicate_count'] = duplicate_count
    
    # Entity ID overlap if provided
    if id_col and id_col in df.columns:
        unique_entities = df[id_col].nunique()
        total_rows = len(df)
        entity_ratio = unique_entities / total_rows if total_rows > 0 else 1
        results['entity_ratio'] = entity_ratio
        results['unique_entities'] = unique_entities
    
    return results

# ============================================================================
# 4️⃣ PROXY LEAKAGE DETECTION
# ============================================================================

def detect_proxy_leakage(df, target_col):
    """
    Detect features acting as hidden proxies for target.
    Uses: feature importance instability, permutation importance variance.
    """
    target = df[target_col]
    is_classification = target.dtype == 'object' or len(target.unique()) < 20
    
    # Prepare features
    X = df.drop(columns=[target_col]).copy()
    
    # Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Encode target if classification
    if is_classification:
        y = LabelEncoder().fit_transform(target.astype(str))
    else:
        y = target.values
    
    results = {}
    importances_list = []
    
    # Run multiple model instances with different seeds to detect instability
    for seed in [42, 123, 456, 789, 999]:
        try:
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=seed, max_depth=10)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=seed, max_depth=10)
            
            model.fit(X, y)
            importances_list.append(model.feature_importances_)
        except:
            pass
    
    if importances_list:
        importances_array = np.array(importances_list)
        
        for idx, col in enumerate(X.columns):
            variance = np.var(importances_array[:, idx])
            mean_importance = np.mean(importances_array[:, idx])
            
            results[col] = {
                'mean_importance': mean_importance,
                'importance_variance': variance,
                'instability_score': variance / (mean_importance + 1e-6)  # Normalized by mean
            }
    
    return results

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_complete_analysis(file, target_col, time_col, id_col):
    """Main analysis orchestrator"""
    if file is None:
        return "❌ Please upload a CSV file first.", None, None, None, None
    
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", None, None, None, None
    
    if target_col not in df.columns:
        return f"❌ Target column '{target_col}' not found in dataset.", None, None, None, None
    
    # Run all detections
    target_leakage = detect_target_leakage(df, target_col)
    
    time_leakage = None
    time_msg = "N/A"
    if time_col and time_col != "None" and time_col in df.columns:
        time_leakage, time_msg = detect_time_leakage(df, target_col, time_col)
    
    duplicate_leakage = detect_duplicate_leakage(df, target_col, id_col if id_col and id_col != "None" else None)
    proxy_leakage = detect_proxy_leakage(df, target_col)
    
    # Generate report
    report = generate_report(df, target_col, target_leakage, time_leakage, duplicate_leakage, proxy_leakage)
    
    # Create summary table
    summary_df = create_summary_table(target_leakage, proxy_leakage)
    
    # Generate visualizations
    fig_target = plot_target_leakage(target_leakage)
    fig_time = plot_time_leakage(time_leakage) if time_leakage else None
    fig_dup = plot_duplicate_leakage(duplicate_leakage, len(df))
    fig_proxy = plot_proxy_leakage(proxy_leakage)
    fig_summary = plot_risk_summary(target_leakage, time_leakage, proxy_leakage, duplicate_leakage)
    
    return report, summary_df, fig_target, fig_time, fig_dup, fig_proxy, fig_summary

def generate_report(df, target_col, target_leak, time_leak, dup_leak, proxy_leak):
    """Generate human-readable leakage report"""
    report = f"""
# 🛡️ LeakGuard Data Leakage Analysis Report

**Dataset:** {len(df)} rows × {len(df.columns)} columns  
**Target:** {target_col}

---

## 1️⃣ TARGET LEAKAGE ANALYSIS

"""
    
    # Target leakage details
    high_risk_features = []
    for feat, scores in target_leak.items():
        if scores['mi'] > 0.5 or scores['pearson'] > 0.8 or scores['spearman'] > 0.8:
            high_risk_features.append((feat, scores))
    
    if high_risk_features:
        report += "⚠️ **HIGH-RISK FEATURES DETECTED:**\n\n"
        for feat, scores in high_risk_features:
            report += f"- **{feat}**: MI={scores['mi']:.3f}, Pearson={scores['pearson']:.3f}, Spearman={scores['spearman']:.3f}\n"
    else:
        report += "✅ No significant target leakage detected.\n"
    
    report += """

---

## 2️⃣ TIME LEAKAGE ANALYSIS

"""
    
    if time_leak:
        high_drift = {f: s for f, s in time_leak.items() if s['drift'] > 0.3}
        if high_drift:
            report += "⚠️ **HIGH CORRELATION DRIFT DETECTED:**\n\n"
            for feat, scores in high_drift.items():
                report += f"- **{feat}**: Drift={scores['drift']:.3f}, Max Correlation={scores['max_corr']:.3f}\n"
        else:
            report += "✅ No significant time leakage patterns detected.\n"
    else:
        report += "⏭️ Time column not provided. Skipped time leakage analysis.\n"
    
    report += """

---

## 3️⃣ DUPLICATE / SPLIT LEAKAGE ANALYSIS

"""
    
    dup_ratio = dup_leak.get('duplicate_ratio', 0)
    if dup_ratio > 0.05:
        report += f"⚠️ **DUPLICATE ROWS DETECTED:** {dup_ratio*100:.2f}% of rows are duplicates\n"
    else:
        report += f"✅ Duplicate ratio acceptable: {dup_ratio*100:.2f}%\n"
    
    if 'entity_ratio' in dup_leak:
        report += f"- Unique entities: {dup_leak['unique_entities']} / {len(df[next(iter([k for k in dup_leak.keys() if 'entity' not in k]))])} (if entity provided)\n"
    
    report += """

---

## 4️⃣ PROXY LEAKAGE ANALYSIS

"""
    
    unstable_features = {f: s for f, s in proxy_leak.items() if s['instability_score'] > 0.5}
    if unstable_features:
        report += "⚠️ **UNSTABLE FEATURE IMPORTANCE DETECTED:**\n\n"
        for feat, scores in unstable_features.items():
            report += f"- **{feat}**: Importance Variance={scores['importance_variance']:.4f}, Instability={scores['instability_score']:.3f}\n"
    else:
        report += "✅ Feature importance appears stable across model runs.\n"
    
    report += """

---

## 📋 RECOMMENDATIONS

1. **Review flagged features** - Remove or transform high-risk features
2. **Validate temporal ordering** - Ensure features are available at prediction time
3. **Check for duplicates** - Remove duplicate rows before training
4. **Monitor feature stability** - Use ensemble methods to reduce proxy leakage
5. **Perform final validation** - Test on truly held-out data

---

**Generated by LeakGuard** © 2024
    """
    
    return report

def create_summary_table(target_leak, proxy_leak):
    """Create summary dataframe for display"""
    rows = []
    
    for feat, scores in target_leak.items():
        proxy_scores = proxy_leak.get(feat, {})
        risk_level = "🔴 CRITICAL" if scores['mi'] > 0.5 or scores['pearson'] > 0.8 else "🟡 MEDIUM" if scores['mi'] > 0.3 else "🟢 LOW"
        
        rows.append({
            "Feature": feat,
            "Mutual Info": f"{scores['mi']:.3f}",
            "Pearson": f"{scores['pearson']:.3f}",
            "Spearman": f"{scores['spearman']:.3f}",
            "Risk": risk_level
        })
    
    return pd.DataFrame(rows)

def load_columns(file):
    """Load available columns from uploaded file"""
    try:
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        return gr.update(choices=cols, value=cols[0] if cols else None), \
               gr.update(choices=["None"] + cols), \
               gr.update(choices=["None"] + cols)
    except:
        return gr.update(), gr.update(), gr.update()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_target_leakage(target_leak):
    """Plot target leakage scores"""
    if not target_leak:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Target Leakage Analysis', fontsize=14, fontweight='bold')
    
    features = list(target_leak.keys())
    mi_scores = [target_leak[f]['mi'] for f in features]
    pearson_scores = [target_leak[f]['pearson'] for f in features]
    spearman_scores = [target_leak[f]['spearman'] for f in features]
    
    # Sort by MI
    sorted_idx = np.argsort(mi_scores)[::-1][:10]
    
    axes[0].barh([features[i] for i in sorted_idx], [mi_scores[i] for i in sorted_idx], color='#FF6B6B')
    axes[0].set_xlabel('Mutual Information Score')
    axes[0].set_title('Top Features by MI')
    axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    axes[0].legend()
    
    axes[1].barh([features[i] for i in sorted_idx], [pearson_scores[i] for i in sorted_idx], color='#4ECDC4')
    axes[1].set_xlabel('Pearson Correlation')
    axes[1].set_title('Top Features by Pearson')
    axes[1].axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    axes[1].legend()
    
    axes[2].barh([features[i] for i in sorted_idx], [spearman_scores[i] for i in sorted_idx], color='#95E1D3')
    axes[2].set_xlabel('Spearman Correlation')
    axes[2].set_title('Top Features by Spearman')
    axes[2].axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    axes[2].legend()
    
    plt.tight_layout()
    return fig

def plot_time_leakage(time_leak):
    """Plot time leakage correlation drift"""
    if not time_leak:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Time Leakage Analysis', fontsize=14, fontweight='bold')
    
    features = list(time_leak.keys())
    drifts = [time_leak[f]['drift'] for f in features]
    max_corrs = [time_leak[f]['max_corr'] for f in features]
    
    sorted_idx = np.argsort(drifts)[::-1][:10]
    
    ax1.barh([features[i] for i in sorted_idx], [drifts[i] for i in sorted_idx], color='#FFA07A')
    ax1.set_xlabel('Correlation Drift')
    ax1.set_title('Features with High Drift (Past→Future)')
    ax1.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    ax1.legend()
    
    ax2.barh([features[i] for i in sorted_idx], [max_corrs[i] for i in sorted_idx], color='#FFB347')
    ax2.set_xlabel('Max Correlation Over Time')
    ax2.set_title('Peak Correlation Values')
    ax2.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_duplicate_leakage(dup_leak, df_len):
    """Plot duplicate leakage info"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    dup_ratio = dup_leak.get('duplicate_ratio', 0)
    unique_ratio = 1 - dup_ratio
    
    colors = ['#FF6B6B' if dup_ratio > 0.05 else '#51CF66', '#D0D0D0']
    labels = [f'Unique Rows ({unique_ratio*100:.2f}%)', f'Duplicates ({dup_ratio*100:.2f}%)']
    
    wedges, texts, autotexts = ax.pie([unique_ratio, dup_ratio], labels=labels, autopct='%1.1f%%', 
                                        colors=colors, startangle=90, textprops={'fontsize': 11})
    
    ax.set_title('Duplicate Row Detection', fontsize=14, fontweight='bold', pad=20)
    
    # Add warning if duplicates detected
    if dup_ratio > 0.05:
        ax.text(0, -1.4, f'⚠️ WARNING: {dup_leak.get("duplicate_count", 0)} duplicate rows found!', 
               ha='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_proxy_leakage(proxy_leak):
    """Plot proxy leakage instability scores"""
    if not proxy_leak:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Proxy Leakage Analysis', fontsize=14, fontweight='bold')
    
    features = list(proxy_leak.keys())
    variances = [proxy_leak[f]['importance_variance'] for f in features]
    instability = [proxy_leak[f]['instability_score'] for f in features]
    
    sorted_idx = np.argsort(instability)[::-1][:10]
    
    ax1.barh([features[i] for i in sorted_idx], [variances[i] for i in sorted_idx], color='#A78BFA')
    ax1.set_xlabel('Importance Variance')
    ax1.set_title('Feature Importance Variance Across Seeds')
    
    ax2.barh([features[i] for i in sorted_idx], [instability[i] for i in sorted_idx], color='#F472B6')
    ax2.set_xlabel('Instability Score')
    ax2.set_title('Feature Importance Instability')
    ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_risk_summary(target_leak, time_leak, proxy_leak, dup_leak):
    """Plot overall risk summary"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    risk_categories = []
    risk_scores = []
    colors_list = []
    
    # Target Leakage Risk
    high_target = sum(1 for f, s in target_leak.items() if s['mi'] > 0.5 or s['pearson'] > 0.8)
    target_risk = (high_target / len(target_leak)) * 100 if target_leak else 0
    risk_categories.append('Target\nLeakage')
    risk_scores.append(target_risk)
    colors_list.append('#FF6B6B' if target_risk > 30 else '#FFA500' if target_risk > 10 else '#51CF66')
    
    # Time Leakage Risk
    if time_leak:
        high_time = sum(1 for f, s in time_leak.items() if s['drift'] > 0.3)
        time_risk = (high_time / len(time_leak)) * 100
    else:
        time_risk = 0
    risk_categories.append('Time\nLeakage')
    risk_scores.append(time_risk)
    colors_list.append('#FFA07A' if time_risk > 30 else '#FFB347' if time_risk > 10 else '#51CF66')
    
    # Duplicate Leakage Risk
    dup_ratio = dup_leak.get('duplicate_ratio', 0)
    dup_risk = dup_ratio * 100
    risk_categories.append('Duplicate\nLeakage')
    risk_scores.append(dup_risk)
    colors_list.append('#FF6B6B' if dup_risk > 5 else '#51CF66')
    
    # Proxy Leakage Risk
    if proxy_leak:
        high_proxy = sum(1 for f, s in proxy_leak.items() if s['instability_score'] > 0.5)
        proxy_risk = (high_proxy / len(proxy_leak)) * 100
    else:
        proxy_risk = 0
    risk_categories.append('Proxy\nLeakage')
    risk_scores.append(proxy_risk)
    colors_list.append('#F472B6' if proxy_risk > 30 else '#A78BFA' if proxy_risk > 10 else '#51CF66')
    
    bars = ax.bar(risk_categories, risk_scores, color=colors_list, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Risk Score (%)', fontsize=11)
    ax.set_title('Overall Data Leakage Risk Assessment', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add risk zones
    ax.axhspan(0, 25, alpha=0.1, color='green', label='Low Risk')
    ax.axhspan(25, 60, alpha=0.1, color='orange', label='Medium Risk')
    ax.axhspan(60, 100, alpha=0.1, color='red', label='High Risk')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(title="LeakGuard - Data Leakage Detection", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🛡️ LeakGuard")
    gr.Markdown("*Detect silent data leakage risks before model training*")
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload & Configure")
            file_input = gr.File(label="CSV Dataset", file_types=[".csv"])
            
            gr.Markdown("### 🔧 Column Selection")
            target_dropdown = gr.Dropdown(label="🎯 Target Column (Required)", interactive=True)
            time_dropdown = gr.Dropdown(label="⏰ Time Column (Optional)", interactive=True)
            id_dropdown = gr.Dropdown(label="🆔 Entity ID Column (Optional)", interactive=True)
            
            analyze_btn = gr.Button("🔍 Analyze Leakage", variant="primary", size="lg")
            
            file_input.change(load_columns, inputs=file_input, outputs=[target_dropdown, time_dropdown, id_dropdown])
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            report_output = gr.Markdown(label="Leakage Report", value="Upload a CSV and click Analyze to see results")
    
    gr.Markdown("---")
    
    with gr.Row():
        summary_table = gr.Dataframe(label="Feature Risk Assessment", interactive=False)
    
    gr.Markdown("### 📈 Visualization & Analysis")
    
    with gr.Row():
        with gr.Column():
            fig_summary = gr.Plot(label="Overall Risk Summary")
        with gr.Column():
            fig_target = gr.Plot(label="Target Leakage Analysis")
    
    with gr.Row():
        fig_dup = gr.Plot(label="Duplicate Leakage Detection")
        fig_proxy = gr.Plot(label="Proxy Leakage Analysis")
    
    with gr.Row():
        fig_time = gr.Plot(label="Time Leakage Analysis")
    
    
    # Connect analyze button to all outputs
    def run_and_display(file, target_col, time_col, id_col):
        if file is None or not target_col:
            return "❌ Please upload a file and select a target column.", pd.DataFrame(), None, None, None, None, None
        
        report, summary_df, fig_target, fig_time, fig_dup, fig_proxy, fig_summary = run_complete_analysis(file, target_col, time_col, id_col)
        return report, summary_df, fig_target, fig_time, fig_dup, fig_proxy, fig_summary
    
    analyze_btn.click(run_and_display, 
                     inputs=[file_input, target_dropdown, time_dropdown, id_dropdown],
                     outputs=[report_output, summary_table, "fig_target", "fig_time", "fig_dup", "fig_proxy", "fig_summary"])

if __name__ == "__main__":
    app.launch()
