# Deeper Experiments: Favoring Structured Kernels

## Problem Statement
Current metrics show FullRank kernel performing as well or better than structured kernels (Periodic/VonMises/Matérn). This is because the current setup has:
- ✅ Lots of data (9 views × ~200 people = ~1800 training samples)
- ✅ All views present during training
- ✅ Interpolation within seen range

→ FullRank can memorize 45 parameters and win

## Goal
Design experiments where structured kernels' inductive biases (smoothness, periodicity) provide clear advantages.

---

## 🥇 Tier 1: EXTREME Regularization Needed
*(Structured kernels will dominate)*

### 1. Hard Held-Out Views ⭐⭐⭐⭐⭐
**Setup**: Train on central views only (-30° to +30°), test on extreme poses (±60°, ±90°)

**Why it works**: 
- FullRank has no inductive bias about angular smoothness
- Must extrapolate far from training distribution
- Periodic/VonMises know angles wrap around smoothly

**Expected winner**: Periodic/VonMises

**Expected performance**:
- Periodic MSE_out: ~0.05
- FullRank MSE_out: ~0.15+ (3× worse!)

**Implementation difficulty**: Easy

**Impact**: 🔥🔥🔥🔥🔥 **HIGHEST**

---

### 2. Few-Shot Learning (Very Few Images per Identity) ⭐⭐⭐⭐⭐
**Setup**: Only 3-5 images per person instead of 9 views

**Why it works**:
- FullRank needs more data to learn 45 parameters
- Periodic/VonMises only have 1 parameter (lengthscale)
- Strong regularization critical with sparse data

**Expected winner**: Periodic/VonMises (only 1 parameter to learn)

**Expected performance**: Periodic wins by 20-40% on MSE_out

**Implementation difficulty**: Medium (need to modify data sampling)

**Impact**: 🔥🔥🔥🔥🔥 **HIGHEST**

---

### 3. Train on Fewer Identities ⭐⭐⭐⭐
**Setup**: Use only 20-50 people instead of all ~200

**Why it works**:
- Less data = stronger need for regularization
- Structured kernels generalize better with limited samples

**Expected winner**: Matérn/Periodic (smooth interpolation helps)

**Expected performance**: Periodic/Matérn win by 10-20% on MSE_out

**Implementation difficulty**: Easy

**Impact**: 🔥🔥🔥🔥 **VERY HIGH**

---

## 🥈 Tier 2: STRONG Regularization Advantage
*(Structured kernels likely win)*

### 4. Cross-Identity View Prediction ⭐⭐⭐⭐
**Setup**: Train on identities A-N, test on new identities M-Z with missing views

**Why it works**:
- Tests if kernel generalizes to unseen people
- View structure should transfer across identities

**Expected winner**: Periodic/VonMises (view structure transfers)

**Implementation difficulty**: Easy (already have train/val split structure)

**Impact**: 🔥🔥🔥🔥 **VERY HIGH** - True out-of-distribution test

---

### 5. Sparse View Sampling (Non-Uniform) ⭐⭐⭐⭐
**Setup**: Train on views [0, 2, 4, 6, 8] (skip every other), predict views [1, 3, 5, 7]

**Why it works**:
- Forces interpolation between distant views (30° gaps)
- Smooth kernels interpolate better

**Expected winner**: Periodic/Matérn (smooth interpolation)

**Implementation difficulty**: Easy

**Impact**: 🔥🔥🔥🔥 **HIGH** - Tests interpolation directly

---

### 6. Added Noise/Corruption ⭐⭐⭐
**Setup**: Add Gaussian noise to images or latent codes during training

**Why it works**:
- Noisy data → need stronger priors
- Structured kernels won't overfit noise patterns

**Expected winner**: Periodic/VonMises (won't overfit noise)

**Implementation difficulty**: Very easy

**Impact**: 🔥🔥🔥 **MEDIUM-HIGH**

---

## 🥉 Tier 3: MODERATE Regularization Advantage
*(Structured kernels should win, but smaller margin)*

### 7. Lower Latent Dimensionality ⭐⭐⭐
**Setup**: Train VAE with zdim=16 instead of zdim=32

**Why it works**:
- Less capacity → need better structure
- Efficient parameterization matters more

**Expected winner**: Periodic (most efficient)

**Implementation difficulty**: Hard (need to retrain VAE)

**Impact**: 🔥🔥🔥 **MEDIUM**

---

### 8. Asymmetric View Distribution ⭐⭐⭐
**Setup**: Train on more left views than right (e.g., 2× more left samples)

**Why it works**:
- Imbalanced data → need to generalize better
- Symmetry assumption helps

**Expected winner**: Periodic/VonMises (symmetry assumption helps)

**Implementation difficulty**: Medium

**Impact**: 🔥🔥🔥 **MEDIUM**

---

### 9. Early Stopping ⭐⭐
**Setup**: Stop training after 20-30 epochs instead of 100

**Why it works**:
- Less time to overfit
- Simpler models learn faster

**Expected winner**: Periodic/VonMises (learn faster with 1 param)

**Implementation difficulty**: Trivial

**Impact**: 🔥🔥 **LOW-MEDIUM**

---

## 📊 RANKED SUMMARY

| Rank | Scenario | Difficulty | Impact | Best Kernel | Expected Advantage |
|------|----------|-----------|--------|-------------|-------------------|
| **1** | Hard held-out views (-30→30 train, ±60/90 test) | Easy | 🔥🔥🔥🔥🔥 | Periodic/VonMises | 3× better MSE_out |
| **2** | Few-shot (3-5 images per identity) | Medium | 🔥🔥🔥🔥🔥 | Periodic/VonMises | 20-40% better |
| **3** | Fewer identities (20-50 people) | Easy | 🔥🔥🔥🔥 | Matérn/Periodic | 10-20% better |
| **4** | Cross-identity (train A-N, test M-Z) | Easy | 🔥🔥🔥🔥 | Periodic/VonMises | 15-30% better |
| **5** | Sparse views (skip every other) | Easy | 🔥🔥🔥🔥 | Periodic/Matérn | 15-25% better |

---

## 💡 Recommended Starting Point

**Start with Experiment #1: Hard Held-Out Views**

**Reasons**:
1. ✅ Easiest to implement (just filter view indices)
2. ✅ Highest expected impact (3× performance difference)
3. ✅ Most interpretable (extrapolation vs interpolation)
4. ✅ Directly tests the core hypothesis (angular smoothness)

---

## 🎯 Core Principle

All successful scenarios share:
```
Less data + Harder task = Need for better inductive bias

FullRank: "I'll memorize the 45 correlations I see"
Periodic: "I know angles are smooth and wrap around"
```

**When you have**:
- ✅ Lots of data per identity (9 views × 200 people)
- ✅ All views present during training
- ✅ Interpolation within seen range
→ **FullRank can memorize and wins**

**When you have**:
- ❌ Sparse views (3-5 instead of 9)
- ❌ Extrapolation to unseen angles
- ❌ Fewer identities
→ **Periodic/VonMises force smooth structure and win decisively!** 🏆

---

## 📝 Notes on Variance Components

The GP model learns two variance components:
- **v₀ (Object Variance)**: How much latent variation comes from object identity
- **v₁ (Noise Variance)**: Unexplained variation (views, noise, other factors)

**Variance Ratio**: `v₀ / (v₀ + v₁)`
- Close to 1.0 → GP successfully learned object structure ✅
- Close to 0.5 → Object and noise equally important
- Close to 0.0 → Model failed to learn structure ❌

Higher v₀ relative to v₁ = Better disentanglement of object identity from other factors.
