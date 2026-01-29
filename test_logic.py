import pandas as pd
import numpy as np
from leakage_detector import LeakageDetector

def test_leakage_detection():
    print("Generating synthetic data...")
    np.random.seed(42)
    n = 1000
    
    # 1. Base Data
    target = np.random.randint(0, 2, n)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'target': target,
        'feature_safe': np.random.randn(n),
        'id': np.arange(n)
    })
    
    # 2. Inject Target Leakage (Direct)
    df['leak_direct'] = df['target'] # Perfect correlation
    
    # 3. Inject Proxy Leakage (Noisy Target)
    df['leak_proxy'] = df['target'] + np.random.normal(0, 0.1, n)
    
    # 4. Inject Time Leakage
    # Feature correlates with target ONLY in the second half (future)
    noise = np.random.randn(n)
    # First half: noise
    # Second half: target + small noise
    half = n // 2
    future_leak = np.concatenate([noise[:half], target[half:] + np.random.normal(0, 0.1, n-half)])
    df['leak_time'] = future_leak
    
    # 5. Inject Duplicates
    # Duplicate first 100 rows and append
    dupes = df.iloc[:100].copy()
    df = pd.concat([df, dupes], ignore_index=True)
    
    print("Running LeakageDetector...")
    detector = LeakageDetector(df, target_col='target', time_col='date', id_col='id', problem_type='classification')
    report = detector.run_all()
    
    print("\n--- TEST RESULTS ---")
    print(f"Overall Score: {report['overall_score']}")
    print(f"Severity: {report['severity']}")
    
    features = report['features']
    
    # Assertions
    failures = []
    
    # Check Direct Leak
    if features['leak_direct']['risk'] != 'Critical':
        failures.append(f"Failed to detect Critical Direct Leak. Got: {features['leak_direct']['risk']}")
    else:
        print("✅ Direct Leak detected")

    # Check Proxy Leak
    if features['leak_proxy']['risk'] not in ['Critical', 'High']:
        failures.append(f"Failed to detect Proxy Leak. Got: {features['leak_proxy']['risk']}")
    else:
         print("✅ Proxy Leak detected")

    # Check Time Leak
    # Time leak logic relies on drift or correlation jump. 
    # The 'leak_time' feature should be flagged.
    # Check reasons for 'Time Leakage' string
    leak_time_reasons = str(features.get('leak_time', {}).get('reasons', []))
    if "Time Leakage" not in leak_time_reasons and features.get('leak_time', {}).get('risk') == 'Low':
         failures.append(f"Failed to detect Time Leakage. Got: {features.get('leak_time')}")
    else:
         print("✅ Time Leak detected")
         
    # Check Duplicate Leakage
    dup_res = report.get("duplicate_leakage", {})
    if dup_res.get("status") != "High Risk":
        failures.append(f"Failed to detect duplicates. Got: {dup_res}")
    else:
        print("✅ Duplicates detected")

    if failures:
        print("\n❌ Failures:")
        for f in failures:
            print(f"- {f}")
    else:
        print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    test_leakage_detection()
