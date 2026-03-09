# GNSS Anti Spoofing Detection 🛰️

This repository contains a machine learning pipeline designed to detect GPS/GNSS spoofing attacks. Instead of relying on deep learning models, this solution leans heavily into **First Principle Reasoning** and physics informed feature engineering to create a robust, explainable, and highly deployable detection system.

## 1. Problem Understanding
A GNSS spoofer attempts to broadcast fake satellite signals to deceive a receiver about its true location or time. While difficult to detect through position alone, spoofing leaves subtle physical and mathematical artifacts in the raw satellite tracking data. 

Because spoofing attacks happen to the entire receiver (not just a single satellite), the core challenge is transforming multi channel chaotic tracking data into a holistic "snapshot" of the sky at any given millisecond, and finding the physical impossibilities within it.

## 2. Feature Engineering
The pipeline avoids raw data dumping and instead calculates domain specific metrics based on the laws of physics and signal processing:

* **SQM (Signal Quality Monitoring):** Measures peak symmetry (`delta_metric`) and energy leakage into the Quadrature phase (`qsqm`). Spoofing attacks inherently distort the correlation peaks.
* **Physics Informed Residuals (PIR):** Calculates Doppler acceleration (`doppler_rate`) and a geometric consistency check (`doppler_pseudo_residual = pseudorange_rate + Carrier_Doppler_hz × λ`). For authentic signals this residual is ~0 (range rate = −Doppler frequency × L1 wavelength). A spoofer cannot maintain this relationship across all channels simultaneously, producing non-zero residuals.
* **Wavelet Spectral Entropy:** Uses Daubechies (`db8`) discrete wavelet transforms to measure the "chaos" of the tracking loops. Fake signals cause the tracking loops to snap, creating high frequency entropy.
* **Cross Channel Consistency:** Spoofers usually broadcast from a single local antenna. By aggregating features across all active channels (e.g., measuring the standard deviation of `CN0` across the sky), the model detects when multiple satellites suddenly suspiciously share the exact same signal characteristics.

## 3. Model Architecture
We use **XGBoost (eXtreme Gradient Boosting)**.

While architectures like Temporal Fusion Transformers (TFT) were considered, XGBoost was selected because:
1. **Explainability:** In aerospace/defense, alarms must be explainable. XGBoost provides explicit feature importance (e.g., "The alarm triggered because the `doppler_pseudo_residual` violated physical limits").
2. **Speed & Deployability:** Tree based models can run inference in microseconds on cheap, low power embedded CPUs (like a drone's flight controller), whereas deep learning requires heavy PyTorch binaries and GPUs.
3. **Robustness to Noise:** XGBoost naturally ignores useless features and prevents overfitting via specific hyperparameters (`colsample_bytree`, `subsample`).

## 4. Training Methodology & Data Leakage Prevention
The model is trained to minimize **LogLoss** but evaluated strictly on **Weighted F1 Score**, because spoofing is a severe class imbalance problem (authentic data heavily outweighs attacked data).

**Strict Data Leakage Controls:**
* **Temporal Block Splitting:** We use `StratifiedGroupKFold` (5 folds) over 20 contiguous temporal blocks. The timeline is divided into sequential chunks and entire blocks are held out as validation, ensuring that rolling/lag features for any training row can only look back into earlier training blocks — never into the validation period. This eliminates the temporal leakage that would occur with a shuffled split. Stratification ensures every fold still sees spoofing examples despite their rarity.
* **Native Class Balancing:** Instead of generating synthetic data via SMOTE (which is prone to distribution leakage), we natively handle class imbalance directly inside the XGBoost algorithm by dynamically calculating `scale_pos_weight` for each training fold. This heavily penalizes the model mathematically if it misses a scarce spoofing attack.
* **Causal Lags:** All rolling window and lag features (`.shift(1)`) are strictly causal — they only look backwards in time within each satellite channel group.
* **Ensemble Inference:** All 5 fold models are saved and used together at inference time. Their `predict_proba` outputs are averaged, which is more robust than any single fold model.
* **Threshold Tuning:** After training, the optimal decision threshold is found by scanning OOF probabilities and maximising F1 for the minority (Spoofed) class. The tuned threshold is saved to `models/threshold.json` and applied automatically during inference, improving recall on attacks without collapsing precision.

## 5. How to Run (Reproducibility)

1. Ensure the raw `train.csv` and `test.csv` files are in the root directory.
2. Install requirements using `pip install -r requirements.txt`
3. Run the complete end-to-end pipeline:
```bash
python src/main.py
```
This script will automatically execute the physics feature extraction (`features.py`), aggregate the data to one row per timestamp (`feature_engineering.py`), train XGBoost with temporal-block cross-validation (`train.py`), and save the best fold's model weights to `models/xgb_model.json`.

4. To generate predictions on the test set:
```bash
python src/inference.py
```
This loads the saved model, runs the same feature pipeline on `test.csv`, and writes `submission.csv`.


