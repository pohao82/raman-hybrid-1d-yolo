"""Streamlit adapter for the Raman analysis application.

Keep Streamlit-specific code here.  The scientific workflow lives in
app.workflow, making a future Dash frontend possible without rewriting the
analysis engine.
"""

import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run app/app.py` from the project root or app directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.interactive_plot import (
    build_detection_figure,
    build_fit_figure,
    classify_click,
    seed_peak_from_click,
)
from app.pipeline import crop_active_fitting_window, preprocess_spectrum
from app.sidebar import SidebarConfig, render_sidebar
from app.workflow import AnalysisState, DET_COLS, FIT_COLS
from processing.plotting import plot_reconstruction


def get_state() -> AnalysisState:
    """Return the framework-independent state stored by Streamlit."""
    if "analysis" not in st.session_state:
        st.session_state.analysis = AnalysisState()
    return st.session_state.analysis


def clear_editor_state() -> None:
    """Clear widget deltas after replacing a table from a plot interaction."""
    for key in ("peaks_editor", "fit_editor", "detect_plot", "fit_plot"):
        st.session_state.pop(key, None)


def render_figure(fig, render_format: str, dpi: int) -> None:
    """Render matplotlib output without blurring the scientific figure."""
    if "SVG" in render_format:
        buffer = io.StringIO()
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
        )
        st.image(buffer.getvalue(), width="stretch")
    else:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        st.image(buffer.getvalue(), width="stretch")

    plt.close(fig)


def configure_page() -> None:
    st.set_page_config(
        page_title="Raman Peak Detection & Refinement",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }
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
        [data-testid="stMetricLabel"] p {
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] div {
            font-size: 1.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_raw_stage(state, ui: SidebarConfig, cfg, resampled_raw):
    raw_freq = ui.raw_freq
    raw_intensity = ui.raw_intensity

    col1, col2, col3 = st.columns(3)
    col1.metric("Experimental Data Points", len(raw_freq))
    col2.metric(
        "Wavenumber Range",
        f"{raw_freq[0]:.1f} – {raw_freq[-1]:.1f} cm⁻¹",
    )
    col3.metric(
        "Peak Intensity (Max)",
        f"{np.nanmax(raw_intensity):.4f}",
    )

    st.info(
        "💡 **Raw experimental spectrum loaded.** "
        "Click **'🔍 Detect Peaks'** in the sidebar to localize peaks with the FCN."
    )

    fig = plot_reconstruction(
        freq=cfg.W,
        raw_signal=resampled_raw,
        fit_result=None,
        predicted_peaks=None,
        raw_freq=raw_freq,
        raw_intensity=raw_intensity,
        show_initial_proxy=False,
        show_predicted_lines=False,
        title="Experimental Raman Spectrum (Raw Measurement)",
        xlim=ui.xlim,
        ylim_main=ui.ylim,
        figsize=(16, 6),
        dpi=ui.fig_dpi,
        show_plot=False,
    )
    render_figure(fig, ui.render_format, ui.fig_dpi)


def detection_signature(ui: SidebarConfig):
    return (
        ui.data_key,
        round(float(ui.conf_threshold), 4),
        bool(ui.use_movavg),
        round(float(ui.idiff_threshold), 4),
        bool(ui.use_flat),
        round(float(ui.flat_threshold), 4),
        bool(ui.apply_savgol),
        int(ui.smooth_window),
        int(ui.smooth_polyorder),
    )


def render_detection_stage(state, ui: SidebarConfig, cfg, resampled_signal, resampled_raw):
    signature = detection_signature(ui)

    with st.spinner("Running FCN peak localization & noise filtering..."):
        state.seed_detection(
            signature=signature,
            model=ui.model,
            resampled_signal=resampled_signal,
            cfg=cfg,
            device=ui.device,
            conf_threshold=ui.conf_threshold,
            use_movavg_filter=ui.use_movavg,
            idiff_threshold=ui.idiff_threshold,
            use_flat_filter=ui.use_flat,
            flat_threshold=ui.flat_threshold,
            resampled_raw_signal=resampled_raw,
        )

    peaks = state.peaks_df
    ymax = float(np.nanmax(resampled_raw)) if resampled_raw.size else 1.0
    gamma_mid = float(np.mean(cfg.gamma_range))
    amplitude_default = round(ymax * 0.3, 3) if np.isfinite(ymax) else 1.0
    position_default = round(float(cfg.W[len(cfg.W) // 2]), 2)

    # Plot renders above the editor; its trace order depends on `edited`, so
    # reserve the slot now and fill it after the editor returns.
    plot_area = st.container()

    edit_col, button_col = st.columns([4, 1])
    with edit_col:
        with st.expander("Peak list", expanded=True):
            edited = st.data_editor(
                peaks,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key="peaks_editor",
                height=210,
                column_config={
                    "A": st.column_config.NumberColumn(
                        "A", min_value=0.0, format="%.4f",
                        default=amplitude_default,
                    ),
                    "pos": st.column_config.NumberColumn(
                        "pos (cm⁻¹)",
                        min_value=float(cfg.W[0]),
                        max_value=float(cfg.W[-1]),
                        format="%.2f",
                        default=position_default,
                    ),
                    "gamma": st.column_config.NumberColumn(
                        "γ",
                        min_value=0.01,
                        format="%.3f",
                        default=round(gamma_mid, 3),
                    ),
                },
            )

    with button_col:
        if st.button(
            "↺ Reset to FCN detections",
            width="stretch",
        ):
            state.detect_sig = None
            clear_editor_state()
            st.rerun()

    valid = edited.dropna(how="any")
    rows = valid[DET_COLS].values.tolist()

    with plot_area:
        fig = build_detection_figure(
            cfg.W,
            resampled_raw,
            rows,
            raw_freq=ui.raw_freq,
            raw_intensity=ui.raw_intensity,
            show_initial_proxy=ui.show_initial,
            xlim=ui.xlim,
            ylim=ui.ylim,
            title=(
                f"FCN Peak Detections ({len(rows)} peaks) — "
                "click spectrum to add · click ▼ to remove · edit the table"
            ),
        )
        event = st.plotly_chart(
            fig,
            key="detect_plot",
            on_select="rerun",
            selection_mode=("points",),
        )

    action = classify_click(
        (event.get("selection") or {}).get("points", [])
        if event else []
    )

    if action and action[0] == "add":
        seed = seed_peak_from_click(
            action[1], action[2], cfg.W, resampled_raw, cfg
        )
        state.apply_detection_table(edited)
        state.add_detection_peak(seed)
        clear_editor_state()
        st.rerun()

    if action and action[0] == "remove":
        if 0 <= action[1] < len(valid):
            state.apply_detection_table(edited)
            state.remove_detection_peak(valid.index[action[1]])
            clear_editor_state()
            st.rerun()


def render_fitted_stage(state, ui: SidebarConfig, cfg, freq_in, raw_signal_in):
    if (
        state.fit_result is None
        or state._refit_from_detected
        or state._refit_from_fitted
    ):
        with st.spinner(
            "Refining peak parameters via non-linear least squares..."
        ):
            fitted = state.prepare_fit(
                freq_in=freq_in,
                raw_signal_in=raw_signal_in,
                cfg=cfg,
                refine_mode=ui.refine_mode,
                pos_window=ui.pos_window,
                width_scale=ui.width_scale,
                target_peaks_per_grp=ui.target_peaks_per_grp,
                sep_factor=ui.sep_factor,
            )

        if not fitted:
            if state.fit_result is None:
                st.warning(
                    "No peaks to fit. Go back to **🔍 Detect Peaks** and add some."
                )
            else:
                st.warning(
                    "Refinement returned no result. "
                    "Check the peak list and try again."
                )
            st.stop()

        clear_editor_state()

    if state.fit_result is None:
        st.warning("No fit result is available.")
        st.stop()

    if state.fit_df.empty:
        state.fit_df = pd.DataFrame(
            state.fit_result["peaks"],
            columns=FIT_COLS,
        ).astype(float)

    raw_plot = raw_signal_in
    freq_plot = freq_in

    ymax = float(np.nanmax(raw_plot)) if raw_plot.size else 1.0
    gamma_mid = float(np.mean(cfg.gamma_range))
    amplitude_default = round(ymax * 0.3, 3) if np.isfinite(ymax) else 1.0
    position_default = round(
        float(freq_plot[len(freq_plot) // 2]), 2
    )

    # Plot renders above the editor (see render_detection_stage).
    plot_area = st.container()

    edit_col, button_col = st.columns([4, 1])
    with edit_col:
        with st.expander("Peak list", expanded=True):
            edited = st.data_editor(
                state.fit_df,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key="fit_editor",
                height=210,
                column_config={
                    "A": st.column_config.NumberColumn(
                        "A", min_value=0.0, format="%.4f",
                        default=amplitude_default,
                    ),
                    "x0": st.column_config.NumberColumn(
                        "x0 (cm⁻¹)",
                        min_value=float(freq_plot[0]),
                        max_value=float(freq_plot[-1]),
                        format="%.2f",
                        default=position_default,
                    ),
                    "sigma": st.column_config.NumberColumn(
                        "σ",
                        min_value=1e-3,
                        format="%.3f",
                        default=round(2.0 * gamma_mid, 3),
                    ),
                    "eta": st.column_config.NumberColumn(
                        "η",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                        default=0.5,
                    ),
                },
            )

    with button_col:
        if st.button(
            "🔁 Re-fit with these peaks",
            type="primary",
            width="stretch",
        ):
            state.apply_fit_table(edited)
            state.request_refit()
            clear_editor_state()
            st.rerun()

    valid = edited.dropna(how="any")
    rows = valid[FIT_COLS].values.tolist()

    residual, rms = state.residual_and_rms(freq_plot, raw_plot)

    st.metric("Fit RMS", f"{rms:.6g}")

    with plot_area:
        fig = build_fit_figure(
            freq_plot,
            raw_plot,
            state.fit_result,
            rows,
            raw_freq=ui.raw_freq,
            raw_intensity=ui.raw_intensity,
            show_fit=ui.show_fit,
            show_components=ui.show_components,
            initial_peaks=state.fit_seed_peaks,
            show_initial_proxy=ui.show_initial,
            xlim=ui.xlim,
            ylim=ui.ylim,
            title=(
                f"Pseudo-Voigt Refinement ({len(rows)} peaks) — "
                "click spectrum to add · click ▼ to remove · "
                "edit the table · then Re-fit"
            ),
        )
        event = st.plotly_chart(
            fig,
            key="fit_plot",
            on_select="rerun",
            selection_mode=("points",),
        )

    action = classify_click(
        (event.get("selection") or {}).get("points", [])
        if event else []
    )

    if action and action[0] == "add":
        amplitude, position, gamma = seed_peak_from_click(
            action[1], action[2], freq_plot, raw_plot, cfg
        )
        state.apply_fit_table(edited)
        state.add_fit_peak((amplitude, position, 2.0 * gamma, 0.5))
        clear_editor_state()
        st.rerun()

    if action and action[0] == "remove":
        if 0 <= action[1] < len(valid):
            state.apply_fit_table(edited)
            state.remove_fit_peak(valid.index[action[1]])
            clear_editor_state()
            st.rerun()

    st.subheader("📊 Refined Peak Parameters")
    st.dataframe(
        state.peak_table(),
        width="stretch",
    )

    csv_data = state.peak_table().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Refined Peak Table (CSV)",
        data=csv_data,
        file_name="refined_raman_peaks.csv",
        mime="text/csv",
    )


def main():
    configure_page()
    state = get_state()
    ui = render_sidebar()

    st.markdown(
        '<div class="main-title">'
        "🔬 Physics-Informed Raman Peak Detection & Refinement"
        "</div>",
        unsafe_allow_html=True,
    )

    resampled_signal, resampled_raw = preprocess_spectrum(
        ui.raw_freq,
        ui.raw_intensity,
        ui.cfg,
        apply_savgol=ui.apply_savgol,
        window_length=ui.smooth_window,
        polyorder=ui.smooth_polyorder,
    )

    freq_in, raw_signal_in = crop_active_fitting_window(
        resampled_raw, ui.cfg
    )

    # Buttons are UI events; the workflow owns the resulting state changes.
    if ui.detect_btn:
        state.request_detection()
    if ui.run_fit_btn:
        state.request_fit()
    if ui.reset_fit_btn:
        state.request_reset()

    if state.stage == "raw":
        render_raw_stage(state, ui, ui.cfg, resampled_raw)
    elif state.stage == "detected":
        render_detection_stage(
            state, ui, ui.cfg, resampled_signal, resampled_raw
        )
    elif state.stage == "fitted":
        # Detection must remain synchronized with the current input before fitting.
        signature = detection_signature(ui)
        with st.spinner("Checking FCN detections..."):
            state.seed_detection(
                signature=signature,
                model=ui.model,
                resampled_signal=resampled_signal,
                cfg=ui.cfg,
                device=ui.device,
                conf_threshold=ui.conf_threshold,
                use_movavg_filter=ui.use_movavg,
                idiff_threshold=ui.idiff_threshold,
                use_flat_filter=ui.use_flat,
                flat_threshold=ui.flat_threshold,
                resampled_raw_signal=resampled_raw,
            )
        render_fitted_stage(
            state, ui, ui.cfg, freq_in, raw_signal_in
        )


if __name__ == "__main__":
    main()
