"""Framework-independent Raman analysis workflow.

This module owns application state and orchestration, but knows nothing about
Streamlit, Dash, or any other UI framework.

A future Dash frontend can call the same methods used by the Streamlit adapter.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.pipeline import (
    build_peak_dataframe,
    compute_fit_residual_and_rms,
    fitted_to_seeds,
    run_fcn_detection,
    run_refinement_pipeline,
)

DET_COLS = ["A", "pos", "gamma"]
FIT_COLS = ["A", "x0", "sigma", "eta"]


@dataclass
class AnalysisState:
    """Mutable state for one Raman analysis session.

    The object is deliberately UI-agnostic.  Streamlit can keep it in
    st.session_state; a Dash app can keep equivalent state server-side or
    serialize the simple fields into dcc.Store.
    """

    stage: str = "raw"
    fit_result: Optional[dict] = None
    detect_sig: Optional[tuple] = None
    n_fcn_detected: int = 0
    peaks_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=DET_COLS)
    )
    fit_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=FIT_COLS)
    )
    fit_seed_peaks: List[Tuple] = field(default_factory=list)
    _refit_from_detected: bool = False
    _refit_from_fitted: bool = False

    def request_detection(self) -> None:
        self.stage = "detected"

    def request_fit(self) -> None:
        self.stage = "fitted"
        self._refit_from_detected = True

    def request_reset(self) -> None:
        self.stage = "raw"

    def detection_is_stale(self, signature: tuple) -> bool:
        return self.detect_sig != signature

    def seed_detection(
        self,
        *,
        signature: tuple,
        model: Any,
        resampled_signal: np.ndarray,
        cfg: Any,
        device: Any,
        conf_threshold: float,
        use_movavg_filter: bool,
        idiff_threshold: float,
        use_flat_filter: bool,
        flat_threshold: float,
        resampled_raw_signal: np.ndarray,
    ) -> bool:
        """Run FCN detection only when its inputs have changed.

        Returns True when a fresh detection was performed.
        """
        if not self.detection_is_stale(signature):
            return False

        detected_raw, filtered = run_fcn_detection(
            model,
            resampled_signal,
            cfg,
            device,
            conf_threshold=conf_threshold,
            use_movavg_filter=use_movavg_filter,
            idiff_threshold=idiff_threshold,
            use_flat_filter=use_flat_filter,
            flat_threshold=flat_threshold,
            resampled_raw_signal=resampled_raw_signal,
        )

        peaks = [
            [float(A), float(pos), float(gamma)]
            for (_conf, A, pos, gamma) in filtered
        ]

        self.n_fcn_detected = len(detected_raw)
        self.peaks_df = pd.DataFrame(peaks, columns=DET_COLS).astype(float)
        self.fit_result = None
        self.fit_df = pd.DataFrame(columns=FIT_COLS)
        self.fit_seed_peaks = []
        self.detect_sig = signature
        return True

    def apply_detection_table(self, edited: pd.DataFrame) -> None:
        self.peaks_df = edited[DET_COLS].copy().reset_index(drop=True)

    def add_detection_peak(self, seed: Tuple[float, float, float]) -> None:
        new = pd.DataFrame([seed], columns=DET_COLS)
        self.peaks_df = pd.concat([self.peaks_df, new], ignore_index=True)

    def remove_detection_peak(self, label) -> None:
        """`label` is a `peaks_df` index label (see caller: the ▼-marker
        click index mapped through the valid-rows frame)."""
        self.peaks_df = self.peaks_df.drop(label).reset_index(drop=True)

    def prepare_fit(
        self,
        *,
        freq_in: np.ndarray,
        raw_signal_in: np.ndarray,
        cfg: Any,
        refine_mode: str,
        pos_window: float,
        width_scale: float,
        target_peaks_per_grp: int,
        sep_factor: float,
    ) -> bool:
        """Run refinement when a fit has been requested.

        Returns True if a fit was executed.
        """
        need_fit = (
            self.fit_result is None
            or self._refit_from_detected
            or self._refit_from_fitted
        )

        if not need_fit:
            return False

        resume = self._refit_from_fitted and not self.fit_df.empty

        if resume:
            seeds = fitted_to_seeds(
                self.fit_df.dropna(how="any")[FIT_COLS].values.tolist()
            )
        else:
            seeds = [
                tuple(row)
                for row in self.peaks_df.dropna(how="any")[DET_COLS]
                .values.tolist()
            ]

        if not seeds:
            return False

        self.fit_seed_peaks = [tuple(seed[:3]) for seed in seeds]

        fit_result = run_refinement_pipeline(
            freq_in,
            raw_signal_in,
            seeds,
            refine_mode=refine_mode,
            pos_window=pos_window,
            width_scale=width_scale,
            target_peaks_per_grp=target_peaks_per_grp,
            sep_factor=sep_factor,
            from_fitted=resume,
        )

        if fit_result is None:
            return False

        self.fit_result = fit_result
        self.fit_df = pd.DataFrame(
            fit_result["peaks"], columns=FIT_COLS
        ).astype(float)

        self._refit_from_detected = False
        self._refit_from_fitted = False
        return True

    def request_refit(self) -> None:
        self.stage = "fitted"
        self._refit_from_fitted = True

    def apply_fit_table(self, edited: pd.DataFrame) -> None:
        self.fit_df = edited[FIT_COLS].copy().reset_index(drop=True)

    def add_fit_peak(self, seed: Tuple[float, float, float, float]) -> None:
        new = pd.DataFrame([seed], columns=FIT_COLS)
        self.fit_df = pd.concat([self.fit_df, new], ignore_index=True)

    def remove_fit_peak(self, label) -> None:
        """`label` is a `fit_df` index label (see `remove_detection_peak`)."""
        self.fit_df = self.fit_df.drop(label).reset_index(drop=True)

    def residual_and_rms(
        self, freq: np.ndarray, raw_signal: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        if self.fit_result is None:
            raise RuntimeError("No fit result is available.")
        return compute_fit_residual_and_rms(
            freq, raw_signal, self.fit_result
        )

    def peak_table(self) -> pd.DataFrame:
        rows = self.fit_df.dropna(how="any")[FIT_COLS].values.tolist()
        return build_peak_dataframe([tuple(row) for row in rows])
