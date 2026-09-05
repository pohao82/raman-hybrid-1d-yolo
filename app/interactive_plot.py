"""
Plotly figure builders for the interactive (click-to-edit) peak stages of app.py.

Each figure carries, in a FIXED trace order, a click-selectable spectrum
(trace 0 -> click adds a peak at the clicked x) and a row of down-triangle peak
markers (trace 1 -> click removes that peak). Every other trace is display-only
(`hoverinfo="skip"`, never part of the selection).

Pure functions, no Streamlit. The app renders with
`st.plotly_chart(fig, on_select="rerun", selection_mode=("points",))` and routes the
returned points through `classify_click`.

`plotly` / `src.lineshapes` are imported lazily so the seed/classify helpers stay
usable (and testable) without those installed.
"""

import numpy as np

SPECTRUM_TRACE = 0   # click -> add a peak at the clicked x
PEAKS_TRACE = 1      # click -> remove the peak at point_number


# --------------------------------------------------------------------------- #
# Pure helpers (no plotly)
# --------------------------------------------------------------------------- #
def seed_peak_from_click(x_click, y_click, freq, signal, cfg, half_window=25.0):
    """Auto-seed (A, pos, gamma) for a peak the user added by clicking the spectrum.

    pos   = x_click
    A     = clicked height above a local baseline (10th percentile of the signal
            within +/- half_window cm^-1 of x_click), floored at a small positive
    gamma = middle of cfg.gamma_range
    """
    freq = np.asarray(freq, dtype=float)
    signal = np.asarray(signal, dtype=float)
    mask = np.abs(freq - x_click) <= half_window
    baseline = float(np.percentile(signal[mask], 10)) if mask.any() else 0.0
    A = max(float(y_click) - baseline, 0.02)
    gamma = float(np.mean(cfg.gamma_range))
    return A, float(x_click), gamma


def classify_click(points):
    """Map a plotly on_select `points` payload to an edit action.

    Returns ("add", x, y) | ("remove", idx) | None.
    """
    if not points:
        return None
    p = points[0]
    curve = p.get("curve_number")
    if curve == SPECTRUM_TRACE and p.get("x") is not None:
        return ("add", float(p["x"]), float(p.get("y", 0.0)))
    if curve == PEAKS_TRACE:
        idx = p.get("point_number", p.get("point_index"))
        if idx is not None:
            return ("remove", int(idx))
    return None


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def _add_edit_traces(fig, freq, signal, marker_x, marker_y, row=None):
    """Traces 0 (spectrum, click=add) and 1 (peak markers, click=remove)."""
    import plotly.graph_objects as go

    kw = {} if row is None else dict(row=row, col=1)
    fig.add_trace(go.Scatter(
        x=freq, y=signal, name="spectrum", mode="lines+markers",
        line=dict(color="#111827", width=1.1),
        marker=dict(size=4, opacity=0),          # invisible but selectable
        hovertemplate="add peak @ %{x:.1f} cm⁻¹<extra></extra>",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=list(marker_x), y=[marker_y] * len(marker_x), name="peaks", mode="markers",
        marker=dict(symbol="triangle-down", size=5, color="#e53e3e",
                    line=dict(width=2, color="#7f1d1d")),
        hovertemplate="remove peak @ %{x:.1f} cm⁻¹<extra></extra>",
    ), **kw)


def _base_layout(fig, title, xlim, ylim, height):
    fig.update_layout(
        title=title, dragmode="pan", clickmode="event+select",
        margin=dict(l=60, r=20, t=50, b=45), height=height,
        showlegend=False, plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    if xlim:
        fig.update_xaxes(range=list(xlim))
    if ylim:
        fig.update_yaxes(range=list(ylim))


def build_detection_figure(freq, resampled_raw, peaks, *, raw_freq=None,
                           raw_intensity=None, show_initial_proxy=False,
                           xlim=None, ylim=None, title="FCN Peak Detections"):
    """Single-panel editable detection view. `peaks` is a list of (A, pos, gamma).

    `show_initial_proxy` overlays the Lorentzian-sum reconstruction of `peaks`
    (everything here is "before fitting" by definition) so the raw FCN guess
    can be compared against the signal ahead of running the optimizer.
    """
    import plotly.graph_objects as go
    from src.lineshapes import lorentzian_np

    freq = np.asarray(freq, dtype=float)
    sig = np.asarray(resampled_raw, dtype=float)
    ymax = float(np.nanmax(sig)) if sig.size else 1.0
    marker_y = 1.08 * ymax if ymax > 0 else 1.0
    pos = [p[1] for p in peaks]

    fig = go.Figure()
    _add_edit_traces(fig, freq, sig, pos, marker_y)
    if raw_freq is not None and raw_intensity is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(raw_freq, float), y=np.asarray(raw_intensity, float),
            name="raw", mode="lines", opacity=0.7,
            line=dict(color="#9ca3af", width=0.8, dash="dot"), hoverinfo="skip",
        ))
    if show_initial_proxy and len(peaks) > 0:
        proxy = np.sum([lorentzian_np(freq, float(A), float(p), float(g))
                        for (A, p, g) in peaks], axis=0)
        fig.add_trace(go.Scatter(
            x=freq, y=proxy, name="Initial FCN Proxy", mode="lines",
            line=dict(color="#2563eb", width=1.3, dash="dash"), hoverinfo="skip",
        ))
    for x0 in pos:
        fig.add_vline(x=x0, line=dict(color="#e53e3e", width=0.8, dash="dot"), opacity=0.3)

    _base_layout(fig, title, xlim, ylim, height=520)
    fig.update_xaxes(title_text="Raman shift (cm⁻¹)")
    fig.update_yaxes(title_text="Intensity (a.u.)")
    return fig


def build_fit_figure(freq, raw_signal, fit_result, edit_peaks, *, raw_freq=None,
                     raw_intensity=None, show_fit=True, show_components=True,
                     initial_peaks=None, show_initial_proxy=False,
                     xlim=None, ylim=None, title="Pseudo-Voigt Refinement"):
    """Two-panel (reconstruction + residual) editable fitted view.

    `edit_peaks` is a list of [A, x0, sigma, eta] -- ▼ markers AND the reconstruction
    curve are drawn from it, so manual edits and click-added peaks show immediately.
    `fit_result["baseline"]` is a (b0, b1) linear-baseline placeholder -- refine /
    refine_grouped always return (0.0, 0.0); this stays wired so a caller-supplied
    background can be dropped in later without touching the plot code.

    `show_fit` toggles the summed reconstruction ("fit") curve; `show_components`
    toggles the per-peak curves. `show_initial_proxy` overlays the Lorentzian-sum
    curve of `initial_peaks` -- the (A, pos, gamma) seeds this fit started from --
    as a "before fitting" reference against the optimized "fit" curve. The residual
    panel is always shown.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from src.lineshapes import pseudo_voigt, lorentzian_np

    freq = np.asarray(freq, dtype=float)
    raw = np.asarray(raw_signal, dtype=float)

    b = fit_result.get("baseline", (0.0, 0.0)) if fit_result else (0.0, 0.0)
    b0, b1 = float(b[0]), float(b[1])
    baseline = b0 + b1 * freq

    comp_curves = [pseudo_voigt(freq, float(A), float(x0), float(s), float(eta))
                   for (A, x0, s, eta) in edit_peaks]
    recon = baseline + (np.sum(comp_curves, axis=0) if comp_curves else np.zeros_like(freq))
    resid = raw - recon

    ymax = float(np.nanmax(raw)) if raw.size else 1.0
    marker_y = 1.08 * ymax if ymax > 0 else 1.0
    marker_x = [p[1] for p in edit_peaks]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05)

    _add_edit_traces(fig, freq, raw, marker_x, marker_y, row=1)
    if raw_freq is not None and raw_intensity is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(raw_freq, float), y=np.asarray(raw_intensity, float),
            name="raw", mode="lines", opacity=0.7,
            line=dict(color="#9ca3af", width=0.8, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)
    if show_initial_proxy and initial_peaks:
        proxy = np.sum([lorentzian_np(freq, float(A), float(p), float(g))
                        for (A, p, g) in initial_peaks], axis=0)
        fig.add_trace(go.Scatter(
            x=freq, y=proxy, name="Initial FCN Proxy", mode="lines",
            line=dict(color="#2563eb", width=1.3, dash="dash"), hoverinfo="skip",
        ), row=1, col=1)
    if show_fit:
        fig.add_trace(go.Scatter(x=freq, y=recon, name="fit", mode="lines",
            line=dict(color="#dc2626", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=baseline, name="baseline", mode="lines",
        line=dict(color="#9ca3af", width=1, dash="dot"), hoverinfo="skip"), row=1, col=1)
    if show_components:
        for i, c in enumerate(comp_curves):
            fig.add_trace(go.Scatter(x=freq, y=baseline + c, name=f"P{i+1}",
                mode="lines", opacity=0.6, line=dict(width=1), hoverinfo="skip"),
                row=1, col=1)

    fig.add_trace(go.Scatter(x=freq, y=resid, name="residual", mode="lines",
        line=dict(color="#111827", width=0.9), hoverinfo="skip"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#dc2626", width=0.8, dash="dash"), row=2, col=1)

    # dashed peak-frequency guides through both panels
    for x0 in marker_x:
        fig.add_vline(x=x0, line=dict(color="#e53e3e", width=0.8, dash="dot"), opacity=0.3)

    fig.update_layout(
        title=title, dragmode="pan", clickmode="event+select",
        margin=dict(l=60, r=20, t=50, b=45), height=660,
        showlegend=False, plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    if xlim:
        fig.update_xaxes(range=list(xlim), row=1, col=1)
        fig.update_xaxes(range=list(xlim), row=2, col=1)
    if ylim:
        fig.update_yaxes(range=list(ylim), row=1, col=1)
    return fig
