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
        return "❌ Please upload a CSV file first.", None, None, None
    
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", None, None, None
    
    if target_col not in df.columns:
        return f"❌ Target column '{target_col}' not found in dataset.", None, None, None
    
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
    
    return report, summary_df, target_leakage, proxy_leakage

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
    
    # Connect analyze button to all outputs
    def run_and_display(file, target_col, time_col, id_col):
        if file is None or not target_col:
            return "❌ Please upload a file and select a target column.", pd.DataFrame()
        
        report, summary_df, _, _ = run_complete_analysis(file, target_col, time_col, id_col)
        return report, summary_df
    
    analyze_btn.click(run_and_display, 
                     inputs=[file_input, target_dropdown, time_dropdown, id_dropdown],
                     outputs=[report_output, summary_table])

if __name__ == "__main__":
    app.launch()
