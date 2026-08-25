# Interactive Streamlit GUI: Raman Peak Detection & Refinement

A web-based interactive application ([`app/app.py`](file:///home/phchang/AI_space_sync/raman_peaks_detection/modular_fcn/app/app.py)) providing visualization, peak localization, and on-demand classical Pseudo-Voigt curve refinement for experimental Raman spectroscopy data.

The architecture cleanly decouples:
- **UI Presentation Layer** (`app/app.py`): Streamlit inputs, widgets, controls, and layout.
- **Inference & Processing Pipeline** (`app/pipeline.py`): Model inference, data preprocessing, filtering, and curve fitting.
- **Unified Plotting Engine** (`processing/plotting.py`): High-quality 1-panel and 2-panel spectrum visualizations.

---

### 1. GUI Layout & Workflow

```
+----------------------------------------------------------------------------------------------------+
|  SIDEBAR CONTROLS (LEFT PANEL)                    MAIN PANEL WORKFLOW                              |
|  1. Model Path (.pt & .json)                      - State 0: Raw Spectrum (Initial Default):       |
|  2. File Upload / Sample Dropdown (CSV/TXT)          * Unaltered loaded experimental spectrum      |
|     - Frequency & Intensity Column Selectors         * No candidate overlays                       |
|  3. FCN Detection & Pre-processing:                                                                |
|     - Savitzky-Golay Window & Poly Order Inputs   - State 1: Peak Detections ([🔍 Detect Peaks]):  |
|     - Confidence & Noise Filter Thresholds           * Candidate vertical markers & proxy overlay  |
|  [🔍 Detect Peaks] [🚀 Refine Fit]                                                                 |
|  [🔄 Reset to Raw Data]                           - State 2: Refined Fit ([🚀 Refine Fit]):        |
|  4. Pseudo-Voigt Refinement Settings                 * Reconstructed fit from optimized params     |
|     - Global vs Regional (Divide & Conquer)          * Linear Baseline & Individual Filled Peaks   |
|     - Search window & width bounds                   * Two-Panel Display with Residuals            |
|  5. Plot & Display Controls                          * Refined Parameter Table & CSV Download      |
|     - Toggle components / predicted lines                                                          |
|     - Custom X-axis & Y-axis overrides                                                             |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Features & Capabilities

### 1. Model Checkpoint Ingestion
* **Custom Paths**: Input custom model weights (`.pt`) and architecture configurations (`.json`).
* **Resource Caching**: Utilizes `@st.cache_resource` to load the PyTorch model once into GPU/CPU memory.

### 2. Flexible Experimental Data Input
* **Upload Mode**: Drop in any `.csv`, `.txt`, `.tsv`, or `.dat` spectrum file.
* **Sample Mode**: Direct dropdown access to bundled experimental datasets (`data/experiment_1/VV.txt`, `data/experiment_1/VH.txt`).
* **Multi-Column Handling**: Automatically detects columns and provides interactive dropdowns to map the frequency/wavenumber axis (x) and intensity axis (y).

### 3. FCN Detection & Configurable Pre-processing
* **Configurable Pre-smoothing (Detection Only)**: User-adjustable Savitzky-Golay integer inputs for **Window Length** (odd integer) and **Polynomial Order**. Pre-smoothing is used **strictly for initial FCN peak localization** to suppress high-frequency noise without distorting physical parameters.
* **Confidence Tuning**: Interactive slider for FCN presence threshold ($p \in [0.10, 0.95]$).
* **Noise Rejection**: Choice between flat intensity thresholding, dual-window Savitzky-Golay difference ($I_{\text{diff}}$), or raw candidate pass-through.
* **On-Demand Peak Detection**: Clicking **`🔍 Detect Peaks`** triggers FCN forward inference and displays peak candidate markers overlaid on the spectrum.

### 4. On-Demand Non-Linear Refinement (Fitted on Direct Raw Signal)
* **Direct Raw Signal Fitting**: When executing curve fitting, the non-linear optimizer fits directly against the **unaltered resampled raw signal** (preserving true physical peak amplitudes and lineshape wings) rather than the pre-smoothed signal.
* **Default Raw View**: Initially shows only loaded experimental data without running neural network inference.
* **On-Click Execution**: Clicking **`🚀 Refine Fit`** runs bounded non-linear least squares optimization.
* **Refinement Modes**:
  * **Global Simultaneous Fit** (`refine`): Fits all detected peaks at once.
  * **Divide-and-Conquer Regional Fit** (`refine_grouped`): Partitions complex spectra into natural islands to eliminate cross-talk.
* **Tunable Search Bounds**: Adjust allowed center movement (`pos_window`), width scaling multiplier (`width_scale`), target peaks per group, and island separation factors.

### 5. Two-Panel Visualization (Matching `processing/plotting.py`)
* **Main Panel (Top)**:
  * Raw experimental points / curve.
  * Reconstructed multi-peak fit (computed from the optimized Pseudo-Voigt parameters).
  * Fitted linear baseline ($b_0 + b_1 x$).
  * Individual shaded Pseudo-Voigt component curves (with mixing fraction $\eta$ annotated).
  * Optional before-fitting FCN proxy overlay (dashed blue) for direct comparison.
  * Vertical dashed reference markers for FCN candidate detections.
* **Residual Panel (Bottom)**:
  * Point-by-point residual error ($y_{\text{raw}} - y_{\text{fit}}$) with a zero-reference line.
* **Interactive Display Controls**:
  * **Rendering Engine Selector**: Choose between **Vector (SVG)** (100% vector paths with infinite sharpness across Retina/4K displays) and **High-DPI Raster (PNG)** (customizable 150–400 DPI).
  * Checkbox toggles to show/hide individual lineshape components, before-fit proxy, or FCN vertical lines.
  * Number inputs to override X-axis range ($x_{\min}, x_{\max}$) and Y-axis limits (both main and residual panels).

### 6. Refined Peak Table & CSV Export
* Interactive table detailing each peak's Center ($x_0$), Amplitude ($A$), FWHM Width ($\sigma$), and Mixing fraction ($\eta$ with Lorentzian vs Gaussian character breakdown).
* One-click download button to export the fitted parameter table as `refined_raman_peaks.csv`.

---

## 3. How to Run

### Method 1: Using the GPU Docker Container (Recommended)

Port `8501` is forwarded in [`docker_torch.sh`](file:///home/phchang/AI_space_sync/raman_peaks_detection/modular_fcn/docker_torch.sh):

```bash
# 1. Launch the Docker container
bash docker_torch.sh

# 2. Inside the container, start the Streamlit app
streamlit run app/app.py
```

### Method 2: Local Python Environment

If running directly on the host machine:

```bash
streamlit run app/app.py
```

### Accessing the Web Application
Open your web browser and navigate to:
```
http://localhost:8501
```
