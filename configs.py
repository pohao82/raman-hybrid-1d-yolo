from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:

    # model parameter
    N_INPUT_CHANNELS: int = 1

    # signal parameters 
    K: int = 21 # max number of slots(peaks)
    n_points: int = 4800 # frequency mesh grid point
    pos_range: tuple = (0, 800.0)
    amp_range: tuple = (0.02, 10.0)  # height of the peaks
    gamma_range: tuple = (1.0, 8.0)  # width

    # poisson noise
    scale: int = 80000
    dark: float = 0.005

    # sandwiched cluster
    window_width_range: tuple = (30,80)
    sandwiched_prob: float = 0.25
    n_peak_max: int = 6

    # model layers — single source of truth, must match DenseDetector's conv stack
    BASE_LAYERS: list = field(default_factory=lambda: [
        {"out_channels": 16,  "kernel_size": 11, "stride": 1},
        {"out_channels": 32,  "kernel_size": 9, "stride": 2},
        {"out_channels": 64,  "kernel_size": 7, "stride": 2},
        {"out_channels": 128, "kernel_size": 5, "stride": 2},
        {"out_channels": 256, "kernel_size": 3, "stride": 2},  # just uncomment to add a layer
    ])

    @property
    def stride(self):
        """Total downsampling factor = product of each layer's stride."""
        total = 1
        for layer in self.BASE_LAYERS:
            total *= layer["stride"]
        return total

    @property
    def W(self): # frequency grid
        return np.linspace(*self.pos_range, self.n_points)

    @property
    def l_grid(self):
        assert self.n_points % self.stride == 0, \
            f"n_points ({self.n_points}) must be divisible by stride ({self.stride})"
        return self.n_points // self.stride

    @property
    # take stride on W (::stride), 
    # ensure will be l_grid frequency points (:l_grid)
    # w_grid is like the starting points of each new grid (each grid contains 'stride' points)
    def w_grid(self):
        return self.W[::self.stride][:self.l_grid]

    @property
    def grid_spacing(self):
        return self.w_grid[1] - self.w_grid[0]


@dataclass
class TrainConfig:
    model_name: str = 'dense_model'
    nbatch: int = 64
    n_train: int = 40000
    n_val: int = 4000
    n_test: int = 4000
    n_epochs: int = 90
    learning_rate: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Hard-data mining & retraining parameters
    hard_data_path: str = None      # Path to hard-mined .npz file (e.g. 'hard_mined_v1.npz'); None for standard base training
    hard_oversample: int = 5        # Oversampling multiplier for hard cases when augmenting training data
    export_hard_data: str = 'hard_mined_v1.npz'    # Optional path to export newly mined test failures as .npz (e.g. 'hard_mined_v1.npz')

