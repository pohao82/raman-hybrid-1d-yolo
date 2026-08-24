from scipy.interpolate import interp1d
import numpy as np
import torch

# load model
import json
from configs import Config
from src.models_peak_predict import DenseDetector


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------


def load_model(model_path, config_path, device=None, mode="eval"):
    """
    Load a trained DenseDetector + its config from disk.
    mode = "eval"(default) or "train"
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path) as f:
        cfg = Config(**json.load(f))
    model = DenseDetector(cfg.N_INPUT_CHANNELS, cfg.BASE_LAYERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

    return model, cfg, device


# --- Inference: threshold + greedy NMS ---
# cpu, pure numpy object
def decode_and_nms(presence, offset, amplitude, gamma, cfg, threshold=0.5, min_sep_ratio=1.0):

    amp_range = cfg.amp_range
    gamma_range = cfg.gamma_range
    w_grid = cfg.w_grid
    grid_spacing = cfg.grid_spacing 

    candidates = []
    for g in range(len(presence)):
        if presence[g] > threshold:
            # unnormolize
            pos = w_grid[g] + offset[g]*grid_spacing
            A = amplitude[g]*(amp_range[1]-amp_range[0]) + amp_range[0]
            gam = gamma[g]*(gamma_range[1]-gamma_range[0]) + gamma_range[0]
            candidates.append((presence[g], A, pos, gam))
    candidates.sort(key=lambda c: -c[0])

    detected = []
    for cand in candidates:
        # min_sep now scales with the STRONGER candidate's own fitted
        # width, instead of one fixed value for every peak regardless of
        # how broad or narrow it actually is
        if all(abs(cand[2]-k[2]) > min_sep_ratio*k[3] for k in detected):
            detected.append(cand)
        #else:
        #    print(f' peaks {cand} removed')
    return detected


def predict_peaks_dense(model, raw_signal, cfg, threshold=0.5, min_sep=5.0, device='cpu'):
    """
    Dense-model equivalent of predict_peaks -- forward pass, decode
    grid cells above threshold, then greedy NMS to collapse duplicates. 
    """
    model.eval()
    with torch.no_grad():

        if cfg.N_INPUT_CHANNELS > 1:
            from src.features import average_features
            stacked = average_features(raw_signal, window=11, polyorder=3, deriv=0)
        else:
            stacked = raw_signal

        raw = torch.as_tensor(stacked[None], dtype=torch.float32, device=device)  # (1, 2, N_POINTS)
        # inference
        presence, offset, amplitude, gamma = model(raw)

    # decode_and_nms takes numpy object.
    detected = decode_and_nms(presence=to_numpy(presence[0]),
                              offset=to_numpy(offset[0]),
                              amplitude=to_numpy(amplitude[0]),
                              gamma=to_numpy(gamma[0]),
                              cfg=cfg,
                              threshold=threshold,
                              #min_sep=min_sep
                              #min_sep_ratio=1.5
                              )
    return detected


def resample_to_training_grid(real_freq, real_intensity, W):
    """Real instrument data almost never shares the training grid's exact
    spacing/range/length. Interpolate onto cfg.W (the same axis the model
    was trained on) before running inference -- required, not optional,
    whenever the input isn't already on that exact grid."""
    interpolator = interp1d(real_freq, real_intensity, kind='linear',
                             bounds_error=False, fill_value=0.0)
    return interpolator(W)   # shape (N_POINTS,), matches training exactly


# copy tensor obj to CPU
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
