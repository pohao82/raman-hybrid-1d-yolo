"""
Streamlit Web Application for Raman Peak Detection & Pseudo-Voigt Refinement.

UI Presentation Layer:
  - Sidebar inputs, configuration, and controls.
  - Interactive plot rendering (via processing.plotting).
  - Peak parameter display tables and CSV export.

To run:
  streamlit run app/app.py
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in sys.path regardless of execution directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Custom pipeline & plotting modules
from app.pipeline import (
    load_detector_model,
    parse_spectrum_file,
    preprocess_spectrum,
    run_fcn_detection,
    crop_active_fitting_window,
    run_refinement_pipeline,
    compute_fit_residual_and_rms,
    build_peak_dataframe,
    fitted_to_seeds,
)
from app.interactive_plot import (
    build_detection_figure,
    build_fit_figure,
    classify_click,
    seed_peak_from_click,
)
from processing.plotting import plot_reconstruction


def render_crisp_figure(fig, fmt: str = "Vector (SVG)", dpi: int = 300):
    """
    Renders a matplotlib figure with zero blurriness in Streamlit.
    Using vector SVG by default guarantees pixel-perfect vector lines and text at any resolution.
    """
    if "SVG" in fmt:
        svg_buf = io.StringIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight", facecolor="white")
        st.image(svg_buf.getvalue(), width='stretch')
    else:
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        st.image(png_buf.getvalue(), width='stretch')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Raman Peak Detection & Refinement",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /*.main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }*/

    /* Remove/reduce default padding at the top of the Streamlit page */
    .block-container {
        padding-top: 2.0rem !important;
        padding-bottom: 0rem !important;
    }

    /* Ensure your custom title div doesn't carry top margin */
    .main-title {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        font-size: 2.2rem;
        font-weight: 700;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }

    <style>
    [data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important; /* Smaller label/title */
    }
    [data-testid="stMetricValue"] div {
        font-size: 1.25rem !important; /* Smaller metric number */
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyTorch DenseDetector model...")
def get_cached_model(model_path: str, config_path: str):
    return load_detector_model(model_path, config_path)


# -----------------------------------------------------------------------------
# Sidebar: Model, Data & Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration & Inputs")

    # --- 1. Model Checkpoint Selection ---
    with st.expander("1. Model & Weights", expanded=True):
        default_model = "saved_models/dense_model.pt"
        default_config = "saved_models/dense_model_config.json"

        model_path_input = st.text_input("Model Weights Path (.pt)", value=default_model)
        config_path_input = st.text_input("Model Config Path (.json)", value=default_config)

        try:
            model, cfg, device = get_cached_model(model_path_input, config_path_input)
            st.success(f"Model loaded ({device}): {cfg.n_points} grid pts, stride {cfg.stride}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    # --- 2. Data Selection / Upload ---
    with st.expander("2. Experimental Data Source", expanded=True):
        data_source = st.radio(
            "Select Data Input Method:",
            ["Sample Data", "Upload File (.csv, .txt)"]
        )

        df_data = None
        data_key = None
        if data_source == "Sample Data":
            sample_options = {
                "Experiment 1: VV (Parallel Polarization)": "data/experiment_1/VV.txt",
                "Experiment 1: VH (Cross Polarization)": "data/experiment_1/VH.txt"
            }
            chosen_sample = st.selectbox("Choose sample spectrum:", list(sample_options.keys()))
            sample_file_path = sample_options[chosen_sample]
            data_key = sample_file_path

            if os.path.exists(sample_file_path):
                with open(sample_file_path, "r") as f:
                    df_data = parse_spectrum_file(f, sample_file_path)
            else:
                st.warning(f"Sample file not found at: {sample_file_path}")
        else:
            uploaded_file = st.file_uploader("Upload Spectrum File", type=["csv", "txt", "tsv", "dat"])
            if uploaded_file is not None:
                df_data = parse_spectrum_file(uploaded_file, uploaded_file.name)
                data_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"

        # Column selection if DataFrame is valid
        if df_data is not None and not df_data.empty:
            cols = list(df_data.columns)
            col_freq = st.selectbox("Frequency / Wavenumber Column (x-axis):", cols, index=0)

            val_cols = [c for c in cols if c != col_freq]
            if val_cols:
                col_intensity = st.selectbox("Intensity Column (y-axis):", val_cols, index=0)
            else:
                col_intensity = col_freq

            raw_freq = pd.to_numeric(df_data[col_freq], errors="coerce").dropna().values
            raw_intensity = pd.to_numeric(df_data[col_intensity], errors="coerce").dropna().values

            # Sort by frequency
            sort_order = np.argsort(raw_freq)
            raw_freq = raw_freq[sort_order]
            raw_intensity = raw_intensity[sort_order]
        else:
            st.info("Please provide valid experimental data to begin.")
            st.stop()

    # --- 3. FCN Detection & Filtering Knobs ---
    with st.expander("3. FCN Peak Detection & Pre-processing", expanded=False):
        conf_threshold = st.slider("FCN Presence Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)

        apply_savgol_smooth = st.checkbox("Pre-smooth raw signal before FCN (Savitzky-Golay)", value=False)
        if apply_savgol_smooth:
            col_w, col_p = st.columns(2)
            smooth_window = col_w.number_input("Window Length", min_value=3, max_value=51, value=3, step=2, help="Must be an odd integer")
            smooth_polyorder = col_p.number_input("Poly Order", min_value=1, max_value=5, value=1, step=1)
        else:
            smooth_window, smooth_polyorder = 5, 1

        st.markdown("**Noise Rejection Filters** — applied in sequence")

        use_movavg_filter = st.checkbox("Multiscale moving-average i_diff", value=False)
        if use_movavg_filter:
            idiff_threshold = st.number_input(
                "i_diff threshold",
                min_value=1.0, max_value=10.0, value=2.0, step=0.5
            )
        else:
            idiff_threshold = 3.0

        use_flat_filter = st.checkbox("Flat intensity threshold", value=True)
        if use_flat_filter:
            flat_threshold = st.slider("Flat threshold (raw intensity)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
        else:
            flat_threshold = 0.10

    # --- Workflow Execution Buttons (Right after Section 3) ---
    st.markdown("##### 🚀 Execution Steps")
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        detect_btn = st.button("🔍 Detect Peaks", width='stretch', help="Run FCN model to detect peak locations.")
    with btn_col2:
        run_fit_btn = st.button("🚀 Refine Fit", type="primary", width='stretch', help="Run non-linear Pseudo-Voigt least squares fitting.")
    
    reset_fit_btn = st.button("🔄 Reset to Raw Data", width='stretch', help="Clear detections and view only raw loaded data.")

    # --- Session state for the interactive (click-to-edit) peak workflow ---
    # peaks_df / fit_df are canonical DataFrames owned here; created by the
    # detection/fit paths, edited in place by st.data_editor, and replaced only by
    # external actions (plot click / reseed / reset).
    st.session_state.setdefault("stage", "raw")
    st.session_state.setdefault("fit_result", None)     # last refinement dict
    st.session_state.setdefault("detect_sig", None)     # signature of detection inputs
    st.session_state.setdefault("n_fcn_detected", 0)    # raw FCN count for the metric

    if detect_btn:
        st.session_state["stage"] = "detected"
    if run_fit_btn:
        st.session_state["stage"] = "fitted"
        st.session_state["_refit_from_detected"] = True   # fit the current edited list
    if reset_fit_btn:
        st.session_state["stage"] = "raw"

    # --- 4. Non-Linear Curve Refinement Settings ---
    with st.expander("4. Pseudo-Voigt Refinement Settings", expanded=False):
        refine_mode = st.radio("Optimization Mode:", ["Global Simultaneous (`refine`)", "Divide-and-Conquer Regional (`refine_grouped`)"], index=1)
        pos_window = st.slider("Allowed Peak Center Shift (+/- cm⁻¹)", min_value=0.1, max_value=5.1, value=0.2, step=0.2)
        width_scale = st.slider("Width Search Multiplier", min_value=1.5, max_value=4.0, value=2.0, step=0.5)

        if "Regional" in refine_mode:
            target_peaks_per_grp = st.slider("Target Peaks per Group", min_value=2, max_value=15, value=5, step=1)
            sep_factor = st.slider("Island Separation Factor (elbow room)", min_value=2.5, max_value=8.0, value=6.0, step=0.5)
        else:
            target_peaks_per_grp, sep_factor = 5, 6.0

    # --- 5. Display & Plotting Controls ---
    with st.expander("5. Plot & Axis Display Controls", expanded=True):
        show_optimized_fit = st.checkbox("Show Optimized Fit (After Fitting)", value=True)
        show_initial_proxy = st.checkbox("Show Initial FCN Proxy (Before Fitting)", value=False)
        show_components = st.checkbox("Show Individual Pseudo-Voigt Peaks (After Fitting)", value=True)
        show_predicted_lines = st.checkbox("Show FCN Peak Center Lines", value=True)
        
        render_format = st.radio(
            "Rendering Engine:",
            ["Vector (SVG) — Infinitely Sharp", "High-DPI Raster (PNG)"],
            index=0,
            help="SVG renders vector graphics for crystal-clear lines at any screen resolution."
        )
        if "PNG" in render_format:
            fig_dpi = st.slider("PNG Resolution (DPI)", min_value=150, max_value=400, value=300, step=50)
        else:
            fig_dpi = 200

        st.subheader("Axis Limits (Leave 0 for Auto)")
        x_min_user = st.number_input("X-min (cm⁻¹)", value=0.0, step=50.0)
        x_max_user = st.number_input("X-max (cm⁻¹)", value=600.0, step=50.0)

        custom_y = st.checkbox("Override Y-Axis Limits", value=False)
        if custom_y:
            y_min_user = st.number_input("Main Panel Y-min", value=-0.10, step=0.1)
            y_max_user = st.number_input("Main Panel Y-max", value=float(np.nanmax(raw_intensity) * 1.1), step=0.5)
            y_res_min = st.number_input("Residual Y-min", value=-0.20, step=0.05)
            y_res_max = st.number_input("Residual Y-max", value=0.40, step=0.05)


# -----------------------------------------------------------------------------
# Main Application Content & Pipeline Execution
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🔬 Physics-Informed Raman Peak Detection & Refinement</div>', unsafe_allow_html=True)

# 1. Preprocess Spectrum (Detection signal vs Raw signal)
resampled_signal, resampled_raw_signal = preprocess_spectrum(
    raw_freq, raw_intensity, cfg,
    apply_savgol=apply_savgol_smooth,
    window_length=smooth_window,
    polyorder=smooth_polyorder
)

# Crop Active Signal Range for Fitting
freq_in, raw_signal_in = crop_active_fitting_window(resampled_raw_signal, cfg)
freq_plot, raw_plot = freq_in, raw_signal_in

# Signature of everything that affects FCN detection + filtering. When it changes,
# the detected-peak list is reseeded from a fresh run (discarding manual edits).
detect_sig = (
    data_key,
    round(float(conf_threshold), 4),
    bool(use_movavg_filter), round(float(idiff_threshold), 4),
    bool(use_flat_filter), round(float(flat_threshold), 4),
    bool(apply_savgol_smooth), int(smooth_window), int(smooth_polyorder),
)


DET_COLS = ["A", "pos", "gamma"]
FIT_COLS = ["A", "x0", "sigma", "eta"]


def _run_detection():
    """Fresh FCN detection + filtering -> (raw count, list of [A, pos, gamma])."""
    detected_raw, filtered = run_fcn_detection(
        model, resampled_signal, cfg, device,
        conf_threshold=conf_threshold,
        use_movavg_filter=use_movavg_filter,
        idiff_threshold=idiff_threshold,
        use_flat_filter=use_flat_filter,
        flat_threshold=flat_threshold,
        resampled_raw_signal=resampled_raw_signal,
    )
    peaks = [[float(A), float(pos), float(g)] for (conf, A, pos, g) in filtered]
    return len(detected_raw), peaks


def _clear_editor_state():
    """Drop the stateful widget deltas so a replaced DataFrame renders cleanly."""
    for k in ("peaks_editor", "fit_editor", "detect_plot", "fit_plot"):
        st.session_state.pop(k, None)


def _seed_detection_if_stale():
    """(Re)seed peaks_df from a fresh FCN run whenever the detection inputs change."""
    if st.session_state["detect_sig"] != detect_sig:
        with st.spinner("Running FCN peak localization & noise filtering..."):
            n_raw, peaks = _run_detection()
        st.session_state["n_fcn_detected"] = n_raw
        st.session_state["peaks_df"] = pd.DataFrame(peaks, columns=DET_COLS).astype(float)
        st.session_state["fit_result"] = None
        st.session_state.pop("fit_df", None)          # stale fit
        st.session_state["detect_sig"] = detect_sig
        _clear_editor_state()


stage = st.session_state["stage"]

if stage == "raw":
    # -------------------------------------------------------------------------
    # STAGE 0: Initial Raw Data View (Default - only experimental curve)
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Experimental Data Points", len(raw_freq))
    col2.metric("Wavenumber Range", f"{raw_freq[0]:.1f} – {raw_freq[-1]:.1f} cm⁻¹")
    col3.metric("Peak Intensity (Max)", f"{np.nanmax(raw_intensity):.4f}")

    st.info("💡 **Raw experimental spectrum loaded.** Click **'🔍 Detect Peaks'** in the sidebar to localize peaks with the FCN.")

    fig = plot_reconstruction(
        freq=cfg.W,
        raw_signal=resampled_raw_signal,
        fit_result=None,
        predicted_peaks=None,
        raw_freq=raw_freq,
        raw_intensity=raw_intensity,
        show_initial_proxy=False,
        show_predicted_lines=False,
        title="Experimental Raman Spectrum (Raw Measurement)",
        xlim=(x_min_user, x_max_user) if (x_min_user != 0 or x_max_user != 0) else None,
        ylim_main=(y_min_user, y_max_user) if custom_y else None,
        figsize=(16, 6),
        dpi=fig_dpi,
        show_plot=False
    )
    render_crisp_figure(fig, fmt=render_format, dpi=fig_dpi)

elif stage == "detected":
    # -------------------------------------------------------------------------
    # STAGE 1: Interactive FCN detections -- click spectrum to add, ▼ to remove
    # -------------------------------------------------------------------------
    _seed_detection_if_stale()

    _xlim = (x_min_user, x_max_user) if (x_min_user != 0 or x_max_user != 0) else None
    _ylim = (y_min_user, y_max_user) if custom_y else None
    _gamma_mid = float(np.mean(cfg.gamma_range))
    _sig_max = float(np.nanmax(resampled_raw_signal)) if resampled_raw_signal.size else 1.0
    _A_def = round(_sig_max * 0.3, 3) if np.isfinite(_sig_max) else 1.0
    _mid = round(float(cfg.W[len(cfg.W) // 2]), 2)

    plot_area = st.container()

    # --- editable table (renders first; its return value is this run's truth) ---
    edit_col, btn_col = st.columns([4, 1])
    with edit_col:
        with st.expander("Peak list", expanded=True):
            edited = st.data_editor(
                st.session_state["peaks_df"], num_rows="dynamic", hide_index=True,
                width='stretch', key="peaks_editor", height=210,
                column_config={
                    "A": st.column_config.NumberColumn("A", min_value=0.0, format="%.4f", default=_A_def),
                    "pos": st.column_config.NumberColumn(
                        "pos (cm⁻¹)", min_value=float(cfg.W[0]), max_value=float(cfg.W[-1]),
                        format="%.2f", default=_mid),
                    "gamma": st.column_config.NumberColumn("γ", min_value=0.01, format="%.3f",
                                                          default=round(_gamma_mid, 3)),
                },
            )
    with btn_col:
        if st.button("↺ Reset to FCN detections", width='stretch'):
            st.session_state["detect_sig"] = None
            _clear_editor_state()
            st.rerun()

    _valid = edited.dropna(how="any")            # complete rows; markers drawn from these
    rows = _valid[DET_COLS].values.tolist()

    with plot_area:
        fig = build_detection_figure(
            cfg.W, resampled_raw_signal, rows,
            raw_freq=raw_freq, raw_intensity=raw_intensity,
            xlim=_xlim, ylim=_ylim,
            title=(f"FCN Peak Detections ({len(rows)} peaks)   —   "
                   "click spectrum to add · click ▼ to remove · or edit the table"),
        )
        event = st.plotly_chart(fig, key="detect_plot", on_select="rerun",
                                selection_mode=("points",))

    act = classify_click((event.get("selection") or {}).get("points", []) if event else [])
    if act and act[0] == "add":
        seed = list(seed_peak_from_click(act[1], act[2], cfg.W, resampled_raw_signal, cfg))
        new = pd.DataFrame([seed], columns=DET_COLS)
        st.session_state["peaks_df"] = pd.concat([edited, new], ignore_index=True)
        _clear_editor_state()
        st.rerun()
    elif act and act[0] == "remove" and 0 <= act[1] < len(_valid):
        st.session_state["peaks_df"] = edited.drop(_valid.index[act[1]]).reset_index(drop=True)
        _clear_editor_state()
        st.rerun()

elif stage == "fitted":
    # -------------------------------------------------------------------------
    # STAGE 2: Interactive Pseudo-Voigt refinement -- edit peaks, then re-fit
    # -------------------------------------------------------------------------
    _seed_detection_if_stale()

    refit_from_detected = st.session_state.pop("_refit_from_detected", False)
    refit_from_fitted = st.session_state.pop("_refit_from_fitted", False)
    need_fit = (st.session_state["fit_result"] is None
                or refit_from_detected or refit_from_fitted)

    if need_fit:
        if refit_from_fitted and "fit_df" in st.session_state:
            seeds = fitted_to_seeds(
                st.session_state["fit_df"].dropna(how="any")[FIT_COLS].values.tolist())
        else:
            seeds = [tuple(r) for r in
                     st.session_state["peaks_df"].dropna(how="any")[DET_COLS].values.tolist()]

        if len(seeds) == 0:
            st.warning("No peaks to fit. Go back to **🔍 Detect Peaks** and add some.")
            st.stop()

        with st.spinner("Refining peak parameters via non-linear least squares..."):
            fr = run_refinement_pipeline(
                freq_in, raw_signal_in, seeds,
                refine_mode=refine_mode,
                pos_window=pos_window,
                width_scale=width_scale,
                target_peaks_per_grp=target_peaks_per_grp,
                sep_factor=sep_factor,
            )
        if fr is None:
            st.warning("Refinement returned no result. Check the peak list and try again.")
            st.stop()
        st.session_state["fit_result"] = fr
        st.session_state["fit_df"] = pd.DataFrame(fr["peaks"], columns=FIT_COLS).astype(float)
        _clear_editor_state()

    fit_result = st.session_state["fit_result"]
    if "fit_df" not in st.session_state:   # defensive: fit_result survived without its frame
        st.session_state["fit_df"] = pd.DataFrame(fit_result["peaks"], columns=FIT_COLS).astype(float)

    _xlim = (x_min_user, x_max_user) if (x_min_user != 0 or x_max_user != 0) else None
    _ylim = (y_min_user, y_max_user) if custom_y else None
    _gamma_mid = float(np.mean(cfg.gamma_range))
    _sig_max = float(np.nanmax(raw_plot)) if raw_plot.size else 1.0
    _A_def = round(_sig_max * 0.3, 3) if np.isfinite(_sig_max) else 1.0
    _mid = round(float(freq_plot[len(freq_plot) // 2]), 2)

    plot_area = st.container()

    edit_col, btn_col = st.columns([4, 1])
    with edit_col:
        with st.expander("Peak list", expanded=True):
            edited = st.data_editor(
                st.session_state["fit_df"], num_rows="dynamic", hide_index=True,
                width='stretch', key="fit_editor", height=210,
                column_config={
                    "A": st.column_config.NumberColumn("A", min_value=0.0, format="%.4f", default=_A_def),
                    "x0": st.column_config.NumberColumn(
                        "x0 (cm⁻¹)", min_value=float(freq_plot[0]), max_value=float(freq_plot[-1]),
                        format="%.2f", default=_mid),
                    "sigma": st.column_config.NumberColumn("σ", min_value=1e-3, format="%.3f",
                                                         default=round(2.0 * _gamma_mid, 3)),
                    "eta": st.column_config.NumberColumn("η", min_value=0.0, max_value=1.0,
                                                        format="%.3f", default=0.5),
                },
            )
    with btn_col:
        if st.button("🔁 Re-fit with these peaks", type="primary", width='stretch'):
            st.session_state["fit_df"] = edited          # persist hand edits
            st.session_state["_refit_from_fitted"] = True
            _clear_editor_state()
            st.rerun()

    _valid = edited.dropna(how="any")
    rows = _valid[FIT_COLS].values.tolist()

    residual, rms = compute_fit_residual_and_rms(freq_plot, raw_plot, fit_result)

    with plot_area:
        fig = build_fit_figure(
            freq_plot, raw_plot, fit_result, rows,
            raw_freq=raw_freq, raw_intensity=raw_intensity,
            show_components=show_components, xlim=_xlim, ylim=_ylim,
            title=(f"Pseudo-Voigt Refinement ({len(rows)} peaks)   —   "
                   "click spectrum to add · click ▼ to remove · edit the table · then Re-fit"),
        )
        event = st.plotly_chart(fig, key="fit_plot", on_select="rerun",
                                selection_mode=("points",))

    act = classify_click((event.get("selection") or {}).get("points", []) if event else [])
    if act and act[0] == "add":
        A_s, pos_s, g_s = seed_peak_from_click(act[1], act[2], freq_plot, raw_plot, cfg)
        new = pd.DataFrame([[A_s, pos_s, 2.0 * g_s, 0.5]], columns=FIT_COLS)
        st.session_state["fit_df"] = pd.concat([edited, new], ignore_index=True)
        _clear_editor_state()
        st.rerun()
    elif act and act[0] == "remove" and 0 <= act[1] < len(_valid):
        st.session_state["fit_df"] = edited.drop(_valid.index[act[1]]).reset_index(drop=True)
        _clear_editor_state()
        st.rerun()

    # Refined peak table & CSV export -- reflects the current (possibly edited) list
    st.subheader("📊 Refined Peak Parameters")
    df_peaks = build_peak_dataframe([tuple(r) for r in rows])
    st.dataframe(df_peaks, width='stretch')
    csv_buffer = df_peaks.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Refined Peak Table (CSV)",
        data=csv_buffer,
        file_name="refined_raman_peaks.csv",
        mime="text/csv",
    )
