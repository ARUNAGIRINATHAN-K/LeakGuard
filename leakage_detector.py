import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import scipy.stats as stats
import warnings

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore")

class LeakageDetector:
    def __init__(self, df: pd.DataFrame, target_col: str, time_col: str = None, id_col: str = None, problem_type: str = "auto"):
        self.df = df.copy()
        self.original_df = df # Keep a copy of original
        self.target_col = target_col
        self.time_col = time_col
        self.id_col = id_col
        
        # Determine problem type if auto
        if problem_type == "auto":
            if self.df[target_col].nunique() < 20 and self.df[target_col].dtype == 'object':
                self.problem_type = "classification"
            elif self.df[target_col].nunique() < 10: # Low cardinality numeric can be classif
                 self.problem_type = "classification"
            else:
                 self.problem_type = "regression"
        else:
            self.problem_type = problem_type

        self.report = {
            "overall_score": 0,
            "severity": "Low",
            "features": {},
            "summary": []
        }
    
    def preprocess(self):
        """
        Basic preprocessing for statistical analysis.
        - Drop high null columns (>90%)
        - Drop constant columns
        - Encode object columns for correlation/MI
        - Handle dates
        """
        # Drop meaningless columns
        self.df = self.df.dropna(axis=1, how='all')
        
        # Handle Date Columns: convert to ordinal timestamp if time_col is used as feature, else drop or keep as is
        if self.time_col and self.time_col in self.df.columns:
            try:
                self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
            except:
                pass 
        
        # Encoder for categorical features
        self.label_encoders = {}
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            if col != self.target_col and col != self.id_col and col != self.time_col:
                le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.df[col] = le.fit_transform(self.df[[col]].astype(str))
        
        # Encode target if classification and object
        if self.problem_type == "classification" and self.df[self.target_col].dtype == 'object':
            le_target = OrdinalEncoder()
            self.df[self.target_col] = le_target.fit_transform(self.df[[self.target_col]])

        # Fill remaining NaNs for calculation (simple fill)
        self.df = self.df.fillna(0)

    def detect_target_leakage(self):
        """
        Detects direct feature-target leakage.
        """
        X = self.df.drop(columns=[self.target_col])
        if self.id_col and self.id_col in X.columns:
            X = X.drop(columns=[self.id_col])
        if self.time_col and self.time_col in X.columns:
            X = X.drop(columns=[self.time_col])
            
        y = self.df[self.target_col]
        
        results = {}

        # 1. Correlation (Pearson/Spearman)
        # Fast and first line of defense
        corrs = X.corrwith(y).abs()
        
        # 2. Mutual Information
        if self.problem_type == "classification":
            mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, discrete_features='auto', random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns)

        # 3. Single Feature Predictiveness with a Tree Stump
        # Good for non-linear leakage check
        tree_scores = {}
        for col in X.columns:
            if self.problem_type == "classification":
                model = DecisionTreeClassifier(max_depth=2, random_state=42)
                score = np.mean(cross_val_score(model, X[[col]], y, cv=3, scoring='roc_auc_ovr_weighted'))
            else:
                model = DecisionTreeRegressor(max_depth=2, random_state=42)
                score = np.mean(cross_val_score(model, X[[col]], y, cv=3, scoring='r2'))
            tree_scores[col] = score

        # Analysis
        for col in X.columns:
            risk_level = "Low"
            reasons = []
            
            # Thresholds
            corr_val = corrs.get(col, 0)
            mi_val = mi_series.get(col, 0)
            tree_score = tree_scores.get(col, 0)

            # Strict rules for leaks
            if corr_val > 0.95:
                risk_level = "Critical"
                reasons.append(f"Near-perfect correlation ({corr_val:.2f})")
            elif corr_val > 0.8:
                risk_level = "High" if risk_level != "Critical" else risk_level
                reasons.append(f"Very high correlation ({corr_val:.2f})")
            
            # MI Thresholds need context, but relative spikes are bad.
            # Assuming widely distributed MI, > 0.5 is often suspicious in localized domains
            if mi_val > 0.6: 
                 risk_level = "High" if risk_level != "Critical" else risk_level
                 reasons.append(f"High Mutual Information ({mi_val:.2f})")

            # Predictive Power
            if self.problem_type == "classification" and tree_score > 0.98:
                 risk_level = "Critical"
                 reasons.append(f"Single feature predicts target with AUC {tree_score:.2f}")
            elif self.problem_type == "regression" and tree_score > 0.98:
                 risk_level = "Critical"
                 reasons.append(f"Single feature predicts target with R2 {tree_score:.2f}")

            results[col] = {
                "corr": corr_val,
                "mi": mi_val,
                "tree_score": tree_score,
                "risk": risk_level,
                "reasons": reasons
            }
            
        self.report["features"] = results

    def detect_duplicate_leakage(self):
        """
        Checks for duplicate rows and entity overlaps.
        """
        # Exact duplicates
        dup_count = self.original_df.duplicated().sum()
        total_count = len(self.original_df)
        dup_ratio = dup_count / total_count if total_count > 0 else 0
        
        self.report["duplicate_leakage"] = {
            "duplicate_count": int(dup_count),
            "duplicate_ratio": float(dup_ratio),
            "status": "Safe"
        }

        if dup_ratio > 0.05: # Warn if > 5% are dupes
             self.report["duplicate_leakage"]["status"] = "High Risk"
             self.report["summary"].append(f"Found {dup_count} ({dup_ratio:.1%}) exact duplicate rows. Potential for train-test contamination.")

        # Entity Overlap checks if id_col provided
        if self.id_col and self.id_col in self.original_df.columns:
             # Simulate a random split
             ids = self.original_df[self.id_col]
             if ids.nunique() < len(ids):
                 # We have repeated IDs.
                 # Check ratio of repeats
                 repeat_ratio = 1 - (ids.nunique() / len(ids))
                 if repeat_ratio > 0.1:
                      self.report["summary"].append(f"Entity ID leakage risk: {self.id_col} is repeated in {repeat_ratio:.1%} of rows. Ensure GroupKFold is used.")

    def detect_time_leakage(self):
        """
        Detects if features act differently in future vs past.
        """
        if not self.time_col or self.time_col not in self.original_df:
            return

        df_sorted = self.df.sort_values(self.time_col)
        mid_point = len(df_sorted) // 2
        past = df_sorted.iloc[:mid_point]
        future = df_sorted.iloc[mid_point:]
        
        # Check target correlation shift
        past_corr = past.drop(columns=[self.target_col, self.time_col]).corrwith(past[self.target_col]).abs()
        future_corr = future.drop(columns=[self.target_col, self.time_col]).corrwith(future[self.target_col]).abs()
        
        drift_warnings = []
        for col in past_corr.index:
            p_val = past_corr.get(col, 0)
            f_val = future_corr.get(col, 0)
            
            # If feature becomes MUCH more correlated in future, it might contain "future info"
            # e.g. "OrderStatus" might get updated after target "IsFraud" is finalized
            if f_val > p_val + 0.3: # Significant jump
                 drift_warnings.append(col)
                 if self.report["features"].get(col):
                     self.report["features"][col]["risk"] = "High"
                     self.report["features"][col]["reasons"].append(f"Time Leakage: Correlation jumped from {p_val:.2f} to {f_val:.2f} in future splits.")

        if drift_warnings:
            self.report["summary"].append(f"Time Leakage detected in {len(drift_warnings)} cols: {', '.join(drift_warnings[:3])}...")

    def detect_proxy_leakage(self):
        """
        Train a model multiple times and check feature importance stability.
        Unstable importance often implies proxy or collinearity issues.
        """
        X = self.df.drop(columns=[self.target_col])
        if self.id_col and self.id_col in X.columns: X = X.drop(columns=[self.id_col])
        if self.time_col and self.time_col in X.columns: X = X.drop(columns=[self.time_col])
        y = self.df[self.target_col]

        # Use Random Forest as generic strong learner
        if self.problem_type == "classification":
            model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
            
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
        
        # Check standard deviation of importance
        for i, col in enumerate(X.columns):
            mean_imp = result.importances_mean[i]
            std_imp = result.importances_std[i]
            
            if mean_imp > 0.05 and std_imp > mean_imp * 0.5:
                # High variance relative to importance
                 if self.report["features"].get(col):
                     current_risk = self.report["features"][col]["risk"]
                     # Don't downgrade risk, only upgrade or append
                     if current_risk == "Low":
                         self.report["features"][col]["risk"] = "Medium"
                     self.report["features"][col]["reasons"].append(f"Proxy Risk: Unstable feature importance (Mean: {mean_imp:.3f} +/- {std_imp:.3f})")

    def run_all(self):
        self.preprocess()
        self.detect_target_leakage()
        self.detect_duplicate_leakage()
        self.detect_time_leakage()
        self.detect_proxy_leakage()
        
        # Calculate overall score
        # Simple heuristic: 100 - penalties
        score = 100
        critical_count = sum(1 for f in self.report["features"].values() if f["risk"] == "Critical")
        high_count = sum(1 for f in self.report["features"].values() if f["risk"] == "High")
        
        score -= (critical_count * 50)
        score -= (high_count * 20)
        
        self.report["overall_score"] = max(0, score)
        
        if score < 40:
            self.report["severity"] = "Critical"
        elif score < 70:
            self.report["severity"] = "High"
        elif score < 90:
            self.report["severity"] = "Medium"
        else:
            self.report["severity"] = "Low"
            
        return self.report
