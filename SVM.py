import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import pandas as pd

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

def svm_reweight_rbf_kernel(
    X1, X2u, p2,
    C=1.0,
    gamma='scale',
    max_iter=50000,
    calib_cv=3,
    clip_prob=1e-6,
    verbose=True,
):
    """
    RBF kernel version - handles non-linear patterns.
    Includes built-in separation diagnostics.
    """
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss
    import matplotlib.pyplot as plt
    
    X1 = np.asarray(X1, dtype=np.int64)
    X2u = np.asarray(X2u, dtype=np.int64)
    p2 = np.asarray(p2, dtype=np.float64)
    
    N1, D = X1.shape
    M = X2u.shape[0]
    
    X = np.vstack([X1, X2u])
    y = np.concatenate([np.zeros(N1, dtype=int), np.ones(M, dtype=int)])
    
    w1 = np.ones(N1, dtype=np.float64)
    w2 = p2 / p2.mean()
    sw = np.concatenate([w1, w2])
    
    pi0 = w1.sum() / sw.sum()
    pi1 = w2.sum() / sw.sum()
    
    configs = [
        # (C, degree, gamma, coef0, description)
        (1.0, 2, 'scale', 0, 'Quadratic, no constant'),
        (1.0, 2, 'scale', 1, 'Quadratic, with constant (RECOMMENDED)'),
        (1.0, 2, 'scale', 10, 'Quadratic, high constant'),
        
        (1.0, 3, 'scale', 1, 'Cubic, with constant'),
        (1.0, 4, 'scale', 1, 'Quartic (degree 4)'),
        
        (0.1, 2, 'scale', 1, 'Low C, quadratic'),
        (10.0, 2, 'scale', 1, 'High C, quadratic'),
        
        (1.0, 2, 0.1, 1, 'Quadratic, manual gamma'),
        (1.0, 3, 0.1, 1, 'Cubic, manual gamma'),
    ]
    config_chosen = configs[4]
    C, degree, gamma, coef0, desc = config_chosen
    print(gamma, coef0, desc)
    assert "quartic" in desc.lower(), "Wrong config"
    # RBF kernel SVM
    #base = SVC(C=C, kernel='poly', degree=degree, gamma=gamma, 
    #                  coef0=coef0, max_iter=50000, probability=False)
    #("cal", CalibratedClassifierCV(base, method="sigmoid", cv=calib_cv))
    clf = Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore")),
        ("svm", SVC(
            C=C, 
             kernel='poly', degree=degree, gamma=gamma,
             coef0=coef0, max_iter=50000, probability=True,
        )),
    ])
    
    if verbose:
        print(f"Fitting RBF SVM with C={C}, gamma={gamma}...")
    
    clf.fit(X, y,) #cal__sample_weight=sw)
    
    # === SEPARATION DIAGNOSTICS ===
    if verbose:
        measure_classifier_separation(clf, X, y, sw, X1, X2u, p2)
    
    # Compute weights
    p = clf.predict_proba(X1)[:, 1]
    p = np.clip(p, clip_prob, 1 - clip_prob)
    
    w = (p / (1 - p)) * (pi0 / pi1)
    w = w / w.mean()
    
    return w, clf


def measure_classifier_separation(clf, X, y, sw, X1, X2u, p2, save_plot=False):
    """
    Comprehensive measurement of classifier's ability to separate classes.
    
    Parameters:
    -----------
    clf : fitted classifier
    X : combined features (X1 + X2u)
    y : combined labels
    sw : sample weights
    X1, X2u, p2 : original data for balance checking
    save_plot : whether to save diagnostic plots
    
    Returns:
    --------
    dict with separation metrics
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 80)
    print("CLASSIFIER SEPARATION DIAGNOSTICS")
    print("=" * 80)
    
    # Get predictions
    probs = clf.predict_proba(X)[:, 1]
    probs_survey = probs[y == 0]
    probs_census = probs[y == 1]
    
    # ========== 1. BASIC SEPARATION METRICS ==========
    print("\n[1] BASIC SEPARATION METRICS")
    print("-" * 80)
    
    survey_mean = probs_survey.mean()
    census_mean = probs_census.mean()
    gap = abs(census_mean - survey_mean)
    prob_std = probs.std()
    
    print(f"Survey mean probability:  {survey_mean:.4f}  {'✓ Good' if survey_mean < 0.4 else '✗ Poor' if survey_mean > 0.6 else '~ Fair'}")
    print(f"Census mean probability:  {census_mean:.4f}  {'✓ Good' if census_mean > 0.6 else '✗ Poor' if census_mean < 0.4 else '~ Fair'}")
    print(f"Probability gap:          {gap:.4f}  {'✓ Excellent' if gap > 0.4 else '✓ Good' if gap > 0.3 else '~ Fair' if gap > 0.2 else '✗ Poor'}")
    print(f"Probability std:          {prob_std:.4f}  {'✓ Good' if prob_std > 0.25 else '~ Fair' if prob_std > 0.15 else '✗ Poor'}")
    
    # ========== 2. OVERLAP ANALYSIS ==========
    print("\n[2] DISTRIBUTION OVERLAP ANALYSIS")
    print("-" * 80)
    
    # Calculate overlap percentage
    survey_above_05 = (probs_survey > 0.5).mean()
    census_below_05 = (probs_census < 0.5).mean()
    
    print(f"Survey points predicted as census (p>0.5):  {survey_above_05:.1%}")
    print(f"Census points predicted as survey (p<0.5):  {census_below_05:.1%}")
    
    # Misclassification at 0.5 threshold
    misclass_rate = ((probs_survey > 0.5).sum() + (probs_census < 0.5).sum()) / len(probs)
    print(f"Misclassification rate (threshold=0.5):     {misclass_rate:.1%}  {'✓ Excellent' if misclass_rate < 0.1 else '✓ Good' if misclass_rate < 0.2 else '~ Fair' if misclass_rate < 0.3 else '✗ Poor'}")
    
    # ========== 3. CALIBRATION QUALITY ==========
    print("\n[3] CALIBRATION QUALITY")
    print("-" * 80)
    
    brier = brier_score_loss(y, probs, sample_weight=sw)
    logloss = log_loss(y, probs, sample_weight=sw)
    auc = roc_auc_score(y, probs, sample_weight=sw)
    
    print(f"Brier score:              {brier:.4f}  {'✓ Excellent' if brier < 0.10 else '✓ Good' if brier < 0.15 else '~ Fair' if brier < 0.20 else '✗ Poor'}")
    print(f"Log loss:                 {logloss:.4f}  {'✓ Excellent' if logloss < 0.30 else '✓ Good' if logloss < 0.50 else '~ Fair' if logloss < 0.70 else '✗ Poor'}")
    print(f"ROC AUC:                  {auc:.4f}  {'✓ Excellent' if auc > 0.90 else '✓ Good' if auc > 0.80 else '~ Fair' if auc > 0.70 else '✗ Poor'}")
    
    # ========== 4. SEPARATION POWER ==========
    print("\n[4] SEPARATION POWER")
    print("-" * 80)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((probs_survey.std()**2 + probs_census.std()**2) / 2)
    cohens_d = gap / pooled_std if pooled_std > 0 else 0
    
    print(f"Cohen's d (effect size):  {cohens_d:.4f}  {'✓ Large' if cohens_d > 0.8 else '~ Medium' if cohens_d > 0.5 else '✗ Small'}")
    
    # Separation Index (custom metric: gap / overlap)
    overlap = min(survey_above_05, census_below_05)
    separation_index = gap / (overlap + 0.01)  # Add small epsilon to avoid division by zero
    
    print(f"Separation index:         {separation_index:.4f}  {'✓ Strong' if separation_index > 5 else '~ Moderate' if separation_index > 2 else '✗ Weak'}")
    
    # ========== 5. PERCENTILE ANALYSIS ==========
    print("\n[5] PERCENTILE ANALYSIS")
    print("-" * 80)
    
    survey_percentiles = np.percentile(probs_survey, [25, 50, 75, 95])
    census_percentiles = np.percentile(probs_census, [5, 25, 50, 75])
    
    print(f"Survey percentiles:  25%={survey_percentiles[0]:.3f}, 50%={survey_percentiles[1]:.3f}, 75%={survey_percentiles[2]:.3f}, 95%={survey_percentiles[3]:.3f}")
    print(f"Census percentiles:  5%={census_percentiles[0]:.3f}, 25%={census_percentiles[1]:.3f}, 50%={census_percentiles[2]:.3f}, 75%={census_percentiles[3]:.3f}")
    
    # Ideal: survey's 95th percentile < census's 5th percentile
    ideal_separation = survey_percentiles[3] < census_percentiles[0]
    print(f"\nIdeal separation (survey 95% < census 5%):  {ideal_separation}  {'✓' if ideal_separation else '✗'}")
    
    # ========== 6. OVERALL ASSESSMENT ==========
    print("\n" + "=" * 80)
    print("OVERALL SEPARATION ASSESSMENT")
    print("=" * 80)
    
    score = 0
    max_score = 0
    
    # Scoring criteria
    criteria = [
        ("Probability gap > 0.3", gap > 0.3, 3),
        ("Probability std > 0.2", prob_std > 0.2, 2),
        ("Survey mean < 0.4", survey_mean < 0.4, 2),
        ("Census mean > 0.6", census_mean > 0.6, 2),
        ("Misclassification < 20%", misclass_rate < 0.2, 2),
        ("Brier score < 0.15", brier < 0.15, 2),
        ("ROC AUC > 0.80", auc > 0.80, 3),
        ("Cohen's d > 0.5", cohens_d > 0.5, 2),
    ]
    
    for criterion, passed, points in criteria:
        max_score += points
        if passed:
            score += points
            status = "✓"
        else:
            status = "✗"
        print(f"{status} {criterion:<40} ({points} pts)")
    
    percentage = score / max_score * 100
    print(f"\nTotal score: {score}/{max_score} ({percentage:.0f}%)")
    
    if percentage >= 80:
        quality = "EXCELLENT"
        emoji = "🎉"
        recommendation = "Classifier has strong separation - proceed with confidence!"
    elif percentage >= 60:
        quality = "GOOD"
        emoji = "✓"
        recommendation = "Classifier has decent separation - should work well."
    elif percentage >= 40:
        quality = "FAIR"
        emoji = "⚠"
        recommendation = "Classifier has moderate separation - consider tuning parameters."
    else:
        quality = "POOR"
        emoji = "✗"
        recommendation = "Classifier has weak separation - try different parameters or kernel."
    
    print(f"\n{emoji} SEPARATION QUALITY: {quality}")
    print(f"   {recommendation}")
    
    # Return metrics as dictionary
    return {
        'survey_mean': survey_mean,
        'census_mean': census_mean,
        'gap': gap,
        'std': prob_std,
        'brier_score': brier,
        'log_loss': logloss,
        'roc_auc': auc,
        'cohens_d': cohens_d,
        'misclassification_rate': misclass_rate,
        'separation_index': separation_index,
        'quality': quality,
        'score': score,
        'max_score': max_score,
        'percentage': percentage,
    }

def weighted_pct_T1(T, w):
    T = np.asarray(T, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(T) & np.isfinite(w) & (w >= 0)
    T, w = T[mask], w[mask]

    if w.sum() == 0:
        raise ValueError("Sum of weights is 0 after masking.")
    return 100.0 * (w @ T) / w.sum()


def compare_with_without_calibration(X1, X2u, p2, C=0.2):
    """
    Compare raw SVM scores vs calibrated probabilities.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    import matplotlib.pyplot as plt
    
    X1 = np.asarray(X1, dtype=np.int64)
    X2u = np.asarray(X2u, dtype=np.int64)
    p2 = np.asarray(p2, dtype=np.float64)
    
    X = np.vstack([X1, X2u])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2u))])
    
    w1 = np.ones(len(X1))
    w2 = p2 / p2.mean()
    sw = np.concatenate([w1, w2])
    
    print("=" * 70)
    print("COMPARING RAW SVM vs CALIBRATED")
    print("=" * 70)
    
    # 1. Raw SVM (no calibration)
    print("\n[1] Raw SVM (no calibration)...")
    oh = OneHotEncoder(handle_unknown="ignore")
    X_encoded = oh.fit_transform(X)
    
    svm = LinearSVC(C=C, max_iter=50000, dual=False)
    svm.fit(X_encoded, y, sample_weight=sw)
    
    decision_vals = svm.decision_function(X_encoded)
    
    # Convert to "probabilities" using simple sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    probs_raw = sigmoid(decision_vals)
    
    print(f"  Survey mean: {probs_raw[y==0].mean():.3f}")
    print(f"  Census mean: {probs_raw[y==1].mean():.3f}")
    print(f"  Gap: {abs(probs_raw[y==1].mean() - probs_raw[y==0].mean()):.3f}")
    print(f"  Std: {probs_raw.std():.3f}")
    
    # 2. With cv=3 calibration (your current setup)
    print("\n[2] With cv=3 calibration (current)...")
    clf_cv3 = Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore")),
        ("cal", CalibratedClassifierCV(LinearSVC(C=C, max_iter=50000, dual=False), 
                                       method="sigmoid", cv=3)),
    ])
    clf_cv3.fit(X, y, cal__sample_weight=sw)
    probs_cv3 = clf_cv3.predict_proba(X)[:, 1]
    
    print(f"  Survey mean: {probs_cv3[y==0].mean():.3f}")
    print(f"  Census mean: {probs_cv3[y==1].mean():.3f}")
    print(f"  Gap: {abs(probs_cv3[y==1].mean() - probs_cv3[y==0].mean()):.3f}")
    print(f"  Std: {probs_cv3.std():.3f}")
    
    # 3. With cv=5 calibration
    print("\n[3] With cv=5 calibration...")
    clf_cv5 = Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore")),
        ("cal", CalibratedClassifierCV(LinearSVC(C=C, max_iter=50000, dual=False), 
                                       method="sigmoid", cv=5)),
    ])
    clf_cv5.fit(X, y, cal__sample_weight=sw)
    probs_cv5 = clf_cv5.predict_proba(X)[:, 1]
    
    print(f"  Survey mean: {probs_cv5[y==0].mean():.3f}")
    print(f"  Census mean: {probs_cv5[y==1].mean():.3f}")
    print(f"  Gap: {abs(probs_cv5[y==1].mean() - probs_cv5[y==0].mean()):.3f}")
    print(f"  Std: {probs_cv5.std():.3f}")
    
    # 4. Isotonic calibration (non-parametric, can be less aggressive)
    print("\n[4] With isotonic calibration (cv=3)...")
    clf_iso = Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore")),
        ("cal", CalibratedClassifierCV(LinearSVC(C=C, max_iter=50000, dual=False), 
                                       method="isotonic", cv=3)),
    ])
    clf_iso.fit(X, y, cal__sample_weight=sw)
    probs_iso = clf_iso.predict_proba(X)[:, 1]
    
    print(f"  Survey mean: {probs_iso[y==0].mean():.3f}")
    print(f"  Census mean: {probs_iso[y==1].mean():.3f}")
    print(f"  Gap: {abs(probs_iso[y==1].mean() - probs_iso[y==0].mean()):.3f}")
    print(f"  Std: {probs_iso.std():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    setups = [
        ('Raw SVM (simple sigmoid)', probs_raw),
        ('Calibrated (cv=3, sigmoid)', probs_cv3),
        ('Calibrated (cv=5, sigmoid)', probs_cv5),
        ('Calibrated (cv=3, isotonic)', probs_iso)
    ]
    
    for idx, (name, probs) in enumerate(setups):
        ax = axes[idx // 2, idx % 2]
        
        ax.hist(probs[y==0], bins=30, alpha=0.6, label='Survey', 
                density=True, color='#ff7f0e', edgecolor='black')
        ax.hist(probs[y==1], bins=30, alpha=0.6, label='Census', 
                density=True, color='#1f77b4', edgecolor='black')
        
        ax.axvline(probs[y==0].mean(), color='#ff7f0e', linestyle='--', linewidth=2)
        ax.axvline(probs[y==1].mean(), color='#1f77b4', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nGap: {abs(probs[y==1].mean() - probs[y==0].mean()):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    
    gaps = {
        'raw': abs(probs_raw[y==1].mean() - probs_raw[y==0].mean()),
        'cv3': abs(probs_cv3[y==1].mean() - probs_cv3[y==0].mean()),
        'cv5': abs(probs_cv5[y==1].mean() - probs_cv5[y==0].mean()),
        'iso': abs(probs_iso[y==1].mean() - probs_iso[y==0].mean())
    }
    
    best = max(gaps.items(), key=lambda x: x[1])
    
    if gaps['raw'] > gaps['cv3'] * 1.5:
        print("❌ Calibration is DESTROYING separation!")
        print("   Raw SVM has much better separation than calibrated version.")
        print("\n   Options:")
        print("   1. SKIP calibration entirely (use raw SVM + simple sigmoid)")
        print("   2. Use less aggressive calibration (cv=5 or cv=10)")
        print("   3. Use isotonic calibration (less parametric)")
    elif best[0] == 'cv3':
        print("✓ Current cv=3 sigmoid calibration is best - keep it!")
    else:
        print(f"✓ Better option found: {best[0]} with gap={best[1]:.3f}")
        if best[0] == 'cv5':
            print("   → Increase cv to 5")
        elif best[0] == 'iso':
            print("   → Use isotonic calibration instead of sigmoid")
        elif best[0] == 'raw':
            print("   → Skip CalibratedClassifierCV, use raw SVM")
    
    print(f"\nSaved comparison plot to 'calibration_comparison.png'")
    
    return {
        'raw': probs_raw,
        'cv3': probs_cv3,
        'cv5': probs_cv5,
        'isotonic': probs_iso,
        'gaps': gaps
    }

def quick_distribution_check(X1, X2u, p2):
    """
    Do survey and census actually differ?
    """
    print("Checking if reweighting is even necessary...\n")
    
    for col in range(X1.shape[1]):
        categories = np.unique(np.concatenate([X1[:, col], X2u[:, col]]))
        
        dist_survey = np.array([
            (X1[:, col] == cat).sum() / len(X1) for cat in categories
        ])
        
        dist_census = np.array([
            (p2 * (X2u[:, col] == cat)).sum() / p2.sum() for cat in categories
        ])
        
        tvd = 0.5 * np.abs(dist_survey - dist_census).sum()
        
        print(f"Variable {col}: TVD = {tvd:.4f}", end="")
        
        if tvd < 0.02:
            print(" ← Virtually identical!")
        elif tvd < 0.05:
            print(" ← Very similar")
        elif tvd < 0.10:
            print(" ← Moderately different")
        else:
            print(" ← Very different")


if __name__ == "__main__":
    ground_truth_path='./data/HouseholdPulse_data/cleaned/ipums_cleaned_combined.csv'
    
    
    gt_df = pd.read_csv(ground_truth_path)
    
    X2u = gt_df.to_numpy()[:,1:]
    p2 = gt_df.to_numpy()[:,0]
    Cs = gt_df.iloc[:,1:].nunique().tolist()
    all_predictions={}
    for week_int in range(22,30):
        week = str(week_int)
        bias_path = './data/HouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv'
        bi_df = pd.read_csv(bias_path)
        bi_df_sampled = bi_df.sample(n=2500)
        X1 = bi_df_sampled.to_numpy()[:,1:]
        Y = bi_df_sampled.to_numpy()[:,0]
        
        #w = svm_reweight_x1_to_x2u(X1, X2u, p2, C=1.0, calib_cv=3, clip_prob=1e-6)
        #prediction = weighted_pct_T1(Y,w)
        #print("week: ", week_int, ":", prediction)
        #diagnose_calibration(X1, X2u, p2, C=1.0)
        # Run the comparison
        #results = compare_with_without_calibration(X1, X2u, p2, C=0.2)
        #quick_distribution_check(X1, X2u, p2)
        w, clf = svm_reweight_rbf_kernel(
                    X1, X2u, p2,
                    C=1.0,
                    gamma='scale',
                    max_iter=100000,
                    calib_cv=3,
                    clip_prob=1e-6,
                    verbose=False,)
        # Weighted mean: sum(w_i * T_i) / sum(w_i)
        weighted_avg = np.sum(w * Y) / np.sum(w)
        print(week_int, weighted_avg)