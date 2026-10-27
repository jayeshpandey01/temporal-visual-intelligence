# M1 Data Audit Report

**Date:** 2026-10-28  
**Dataset:** Something-Something V2 (SSv2) Directionally Paired Subset  
**Total Sequences:** 2,000  
**Total Action Classes:** 20 (Balanced: 100 sequences/class)  
**Frames per Sequence:** 8 frames (uniform duration coverage)  
**Resolution:** $224 \times 224 \times 3$ RGB  

---

## 1. 7-Check Integrity Audit Summary

| Check ID | Integrity Dimension | Measured Metric | Threshold / Target | Status |
|:---|:---|:---:|:---:|:---:|
| **Check 1** | Missing Frame Continuity | 0 gaps detected | $t_{k+1} - t_k = 1$ | 🟢 **PASS** |
| **Check 2** | Duplicate Frame Redundancy | Consecutive MSE > 0.001 | No freeze-frame stalls | 🟢 **PASS** |
| **Check 3** | Natural Index Ordering | Monotonic natural sort | No lexical sorting bugs | 🟢 **PASS** |
| **Check 4** | Image Decode Health | 0 corrupt headers | 100% PIL decode validity | 🟢 **PASS** |
| **Check 5** | Spatial Dimension Consistency | (224, 224) uniform | Single fixed aspect ratio | 🟢 **PASS** |
| **Check 6** | Label Completeness & Mapping | 20 unique classes | Bijective integer mapping | 🟢 **PASS** |
| **Check 7** | Class Imbalance Ratio | $R = 1.00\times$ | $R \le 2.0\times$ | 🟢 **PASS** |

---

## 2. Leakage Guard Cross-Split Isolation Verification

Deterministic `GroupKFold` partitioning was executed on `video_id` using fixed seed `42`:
- **Train Split:** 1,400 sequences (70.0%)
- **Validation Split:** 300 sequences (15.0%)
- **Test Split:** 300 sequences (15.0%) — **LOCKED GATE**

```json
{
  "leakage_guard_isolation": {
    "train_val_overlap": 0,
    "train_test_overlap": 0,
    "val_test_overlap": 0,
    "status": "PASS"
  }
}
```
**Conclusion:** Mathematical zero frame leakage verified. Temporally adjacent frames and recording captures never bridge evaluation splits.

---

## 3. Visual Ambiguity & Temporal Strip Analysis

Temporal frame strips were extracted and compiled into `results/figures/audit_strips.png`. Visual inspection confirms:
1. **Single-Frame Ambiguity:** In Category 1 classes (e.g., *Moving away* vs. *Moving towards*, *Pushing left-to-right* vs. *Pushing right-to-left*), Frame 1 displays an object at neutral rest. The static spatial appearance is identical across paired classes.
2. **Dynamic Disambiguation:** From Frame 4 onwards ($t_{\text{informative}}$), the object's spatial trajectory reveals the true action verb.
3. **M1 Gate Decision:** 🟢 **GO** — The dataset is certified for frozen embedding extraction in Milestone 2.

