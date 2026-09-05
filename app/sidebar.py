"""Streamlit sidebar for the Raman analysis app.

Builds the five control sections and returns a single typed `SidebarConfig`
so the stage renderers in `app.app` read `ui.pos_window` instead of a
stringly-typed `ui["pos_window"]`.  Streamlit-only; the scientific layers
(`app.workflow`, `app.pipeline`) do not import this.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.pipeline import load_detector_model, parse_spectrum_file


@dataclass
class SidebarConfig:
    """Everything the main page needs from the sidebar, in one typed object."""

    # model / data
    model: Any
    cfg: Any
    device: Any
    data_key: Optional[str]
    raw_freq: np.ndarray
    raw_intensity: np.ndarray

    # detection + pre-processing knobs
    conf_threshold: float
    apply_savgol: bool
    smooth_window: int
    smooth_polyorder: int
    use_movavg: bool
    idiff_threshold: float
    use_flat: bool
    flat_threshold: float

    # refinement knobs
    refine_mode: str
    pos_window: float
    width_scale: float
    target_peaks_per_grp: int
    sep_factor: float

    # display knobs
    show_fit: bool
    show_initial: bool
    show_components: bool
    show_predicted_lines: bool
    render_format: str
    fig_dpi: int
    xlim: Optional[Tuple[float, float]]
    ylim: Optional[Tuple[float, float]]

    # workflow buttons (this run's click state)
    detect_btn: bool
    run_fit_btn: bool
    reset_fit_btn: bool


@st.cache_resource(show_spinner="Loading PyTorch DenseDetector model...")
def get_cached_model(model_path: str, config_path: str):
    return load_detector_model(model_path, config_path)


def _sidebar_model():
    """Expander 1 -> (model, cfg, device).  Stops the run on load failure."""
    with st.expander("1. Model & Weights", expanded=True):
        default_model = "saved_models/dense_model.pt"
        default_config = "saved_models/dense_model_config.json"

        model_path = st.text_input(
            "Model Weights Path (.pt)", value=default_model
        )
        config_path = st.text_input(
            "Model Config Path (.json)", value=default_config
        )

        try:
            model, cfg, device = get_cached_model(model_path, config_path)
            st.success(
                f"Model loaded ({device}): "
                f"{cfg.n_points} grid pts, stride {cfg.stride}"
            )
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            st.stop()

    return model, cfg, device


def _sidebar_data():
    """Expander 2 -> (data_key, raw_freq, raw_intensity).  Stops on no data."""
    with st.expander("2. Experimental Data Source", expanded=True):
        data_source = st.radio(
            "Select Data Input Method:",
            ["Sample Data", "Upload File (.csv, .txt)"],
        )

        df_data = None
        data_key = None

        if data_source == "Sample Data":
            sample_options = {
                "Experiment 1: VV (Parallel Polarization)": "data/experiment_1/VV.txt",
                "Experiment 1: VH (Cross Polarization)": "data/experiment_1/VH.txt",
            }
            chosen_sample = st.selectbox(
                "Choose sample spectrum:",
                list(sample_options.keys()),
            )
            sample_file = sample_options[chosen_sample]
            data_key = sample_file

            if os.path.exists(sample_file):
                with open(sample_file, "r") as handle:
                    df_data = parse_spectrum_file(handle, sample_file)
            else:
                st.warning(f"Sample file not found at: {sample_file}")
        else:
            uploaded_file = st.file_uploader(
                "Upload Spectrum File",
                type=["csv", "txt", "tsv", "dat"],
            )
            if uploaded_file is not None:
                df_data = parse_spectrum_file(
                    uploaded_file, uploaded_file.name
                )
                data_key = (
                    f"{uploaded_file.name}:"
                    f"{getattr(uploaded_file, 'size', 0)}"
                )

        if df_data is None or df_data.empty:
            st.info("Please provide valid experimental data to begin.")
            st.stop()

        columns = list(df_data.columns)
        col_freq = st.selectbox(
            "Frequency / Wavenumber Column (x-axis):",
            columns,
            index=0,
        )

        value_columns = [c for c in columns if c != col_freq]
        col_intensity = (
            st.selectbox(
                "Intensity Column (y-axis):",
                value_columns,
                index=0,
            )
            if value_columns
            else col_freq
        )

        raw_freq = pd.to_numeric(
            df_data[col_freq], errors="coerce"
        ).dropna().values
        raw_intensity = pd.to_numeric(
            df_data[col_intensity], errors="coerce"
        ).dropna().values

        sort_order = np.argsort(raw_freq)
        raw_freq = raw_freq[sort_order]
        raw_intensity = raw_intensity[sort_order]

    return data_key, raw_freq, raw_intensity


def _sidebar_detection() -> dict:
    """Expander 3 -> detection + pre-processing knobs."""
    with st.expander("3. FCN Peak Detection & Pre-processing", expanded=False):
        conf_threshold = st.slider(
            "FCN Presence Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )

        apply_savgol = st.checkbox(
            "Pre-smooth raw signal before FCN (Savitzky-Golay)",
            value=False,
        )

        if apply_savgol:
            col_w, col_p = st.columns(2)
            smooth_window = col_w.number_input(
                "Window Length",
                min_value=3,
                max_value=51,
                value=3,
                step=2,
            )
            smooth_polyorder = col_p.number_input(
                "Poly Order",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
            )
        else:
            smooth_window, smooth_polyorder = 5, 1

        st.markdown("**Noise Rejection Filters** — applied in sequence")

        use_movavg = st.checkbox(
            "Multiscale moving-average i_diff",
            value=False,
        )
        idiff_threshold = (
            st.number_input(
                "i_diff threshold",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
            )
            if use_movavg
            else 3.0
        )

        use_flat = st.checkbox(
            "Flat intensity threshold",
            value=True,
        )
        flat_threshold = (
            st.slider(
                "Flat threshold (raw intensity)",
                min_value=0.0,
                max_value=0.5,
                value=0.10,
                step=0.01,
            )
            if use_flat
            else 0.10
        )

    return {
        "conf_threshold": conf_threshold,
        "apply_savgol": apply_savgol,
        "smooth_window": smooth_window,
        "smooth_polyorder": smooth_polyorder,
        "use_movavg": use_movavg,
        "idiff_threshold": idiff_threshold,
        "use_flat": use_flat,
        "flat_threshold": flat_threshold,
    }


def _sidebar_actions() -> dict:
    """The three workflow buttons (rendered between sections 3 and 4)."""
    detect_clicked, fit_clicked = st.columns(2)
    with detect_clicked:
        detect_btn = st.button("🔍 Detect Peaks", width="stretch")
    with fit_clicked:
        run_fit_btn = st.button(
            "🚀 Refine Fit", type="primary", width="stretch"
        )

    reset_fit_btn = st.button("🔄 Reset to Raw Data", width="stretch")

    return {
        "detect_btn": detect_btn,
        "run_fit_btn": run_fit_btn,
        "reset_fit_btn": reset_fit_btn,
    }


def _sidebar_refinement() -> dict:
    """Expander 4 -> Pseudo-Voigt refinement knobs."""
    with st.expander("4. Pseudo-Voigt Refinement Settings", expanded=False):
        refine_mode = st.radio(
            "Optimization Mode:",
            [
                "Global Simultaneous (`refine`)",
                "Divide-and-Conquer Regional (`refine_grouped`)",
            ],
            index=1,
        )
        pos_window = st.slider(
            "Allowed Peak Center Shift (+/- cm⁻¹)",
            min_value=0.1,
            max_value=5.1,
            value=0.2,
            step=0.2,
        )
        width_scale = st.slider(
            "Width Search Multiplier",
            min_value=1.5,
            max_value=4.0,
            value=2.0,
            step=0.5,
        )

        if "Regional" in refine_mode:
            target_peaks_per_grp = st.slider(
                "Target Peaks per Group",
                min_value=2,
                max_value=15,
                value=5,
                step=1,
            )
            sep_factor = st.slider(
                "Island Separation Factor (elbow room)",
                min_value=2.5,
                max_value=8.0,
                value=6.0,
                step=0.5,
            )
        else:
            target_peaks_per_grp, sep_factor = 5, 6.0

    return {
        "refine_mode": refine_mode,
        "pos_window": pos_window,
        "width_scale": width_scale,
        "target_peaks_per_grp": target_peaks_per_grp,
        "sep_factor": sep_factor,
    }


def _sidebar_display(raw_intensity: np.ndarray) -> dict:
    """Expander 5 -> plot / axis display knobs (needs raw_intensity for the
    y-max default)."""
    with st.expander("5. Plot & Axis Display Controls", expanded=True):
        show_fit = st.checkbox(
            "Show Optimized Fit (After Fitting)",
            value=True,
        )
        show_initial = st.checkbox(
            "Show Initial FCN Proxy (Before Fitting)",
            value=False,
        )
        show_components = st.checkbox(
            "Show Individual Pseudo-Voigt Peaks (After Fitting)",
            value=True,
        )
        show_predicted_lines = st.checkbox(
            "Show FCN Peak Center Lines",
            value=True,
        )

        render_format = st.radio(
            "Rendering Engine:",
            [
                "Vector (SVG) — Infinitely Sharp",
                "High-DPI Raster (PNG)",
            ],
            index=0,
        )
        fig_dpi = (
            st.slider(
                "PNG Resolution (DPI)",
                min_value=150,
                max_value=400,
                value=300,
                step=50,
            )
            if "PNG" in render_format
            else 200
        )

        st.subheader("Axis Limits (Leave 0 for Auto)")
        x_min = st.number_input("X-min (cm⁻¹)", value=0.0, step=50.0)
        x_max = st.number_input("X-max (cm⁻¹)", value=600.0, step=50.0)

        custom_y = st.checkbox("Override Y-Axis Limits", value=False)
        if custom_y:
            y_min = st.number_input("Main Panel Y-min", value=-0.10, step=0.1)
            y_max = st.number_input(
                "Main Panel Y-max",
                value=float(np.nanmax(raw_intensity) * 1.1),
                step=0.5,
            )
            st.number_input("Residual Y-min", value=-0.20, step=0.05)
            st.number_input("Residual Y-max", value=0.40, step=0.05)
        else:
            y_min = y_max = None

    return {
        "show_fit": show_fit,
        "show_initial": show_initial,
        "show_components": show_components,
        "show_predicted_lines": show_predicted_lines,
        "render_format": render_format,
        "fig_dpi": fig_dpi,
        "xlim": (x_min, x_max) if (x_min != 0 or x_max != 0) else None,
        "ylim": (y_min, y_max) if custom_y else None,
    }


def render_sidebar() -> SidebarConfig:
    """Build every control section and collect them into one `SidebarConfig`."""
    with st.sidebar:
        st.header("⚙️ Configuration & Inputs")

        model, cfg, device = _sidebar_model()
        data_key, raw_freq, raw_intensity = _sidebar_data()

        fields: dict = {}
        fields.update(_sidebar_detection())
        fields.update(_sidebar_actions())
        fields.update(_sidebar_refinement())
        fields.update(_sidebar_display(raw_intensity))

    return SidebarConfig(
        model=model,
        cfg=cfg,
        device=device,
        data_key=data_key,
        raw_freq=raw_freq,
        raw_intensity=raw_intensity,
        **fields,
    )
