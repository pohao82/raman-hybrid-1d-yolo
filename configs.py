from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:

    # model parameter
    N_INPUT_CHANNELS: int = 1

    # signal parameters 
    K: int = 28 # max number of slots(peaks)
    n_points: int = 4800 # frequency mesh grid point
    pos_range: tuple = (0, 800.0)
    amp_range: tuple = (0.02, 14.0)  # height of the peaks
    gamma_range: tuple = (1.0, 8.0)  # width

    # poisson noise
    scale: int = 5000
    dark: float = 0.02

    # sandwiched cluster
    window_width_range: tuple = (30,80)
    n_peak_max: int = 6

    # dataset composition -- fraction of each pattern block in a generated set
    # (see src.signal_sample_module.PATTERN_GENERATORS). Fractions are
    # normalized; add a key here + a generator in the registry to extend.
    dataset_mix: dict = field(default_factory=lambda: {
        "random": 0.45,
        "sandwiched": 0.30,
        "skewed": 0.45,
    })

    # "skewed" pattern: amplitude distribution pushed hard toward small peaks
    skew_small_range: tuple = (0.3, 2.0)
    skew_large_range: tuple = (4.0, 12.0)
    skew_large_prob: float = 0.05

    # model layers — single source of truth, must match DenseDetector's conv stack
    BASE_LAYERS: list = field(default_factory=lambda: [
        {"out_channels": 32,  "kernel_size": 15, "stride": 1},
        {"out_channels": 64,  "kernel_size": 11, "stride": 2},
        {"out_channels": 128, "kernel_size": 9, "stride": 2},
        {"out_channels": 256, "kernel_size": 7, "stride": 2},
        {"out_channels": 512, "kernel_size": 5, "stride": 2},
    ])
        #{"out_channels": 16,  "kernel_size": 11, "stride": 1},
        #{"out_channels": 32,  "kernel_size": 9, "stride": 2},
        #{"out_channels": 64,  "kernel_size": 7, "stride": 2},
        #{"out_channels": 128, "kernel_size": 5, "stride": 2},
        #{"out_channels": 256, "kernel_size": 3, "stride": 2},

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
    n_train: int = 50000
    n_val: int = 5000
    n_test: int = 4000
    n_epochs: int = 140
    learning_rate: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Convergence-based early stop: halt once the validation loss has flattened,
    # measured as the relative change between the mean of the last
    # `convergence_window` epochs and the mean of the `convergence_window`
    # epochs before that. Set convergence_rel_tol = 0.0 to disable.
    convergence_window: int = 10       # epochs per comparison window
    convergence_rel_tol: float = 1e-4  # stop when relative change < this; 0.0 disables
    convergence_min_epochs: int = 40   # never stop before this

    # Hard-data mining & retraining parameters
    hard_data_path: str = None      # Path to hard-mined .npz file (e.g. 'hard_mined_v1.npz'); None for standard base training
    hard_oversample: int = 5        # Oversampling multiplier for hard cases when augmenting training data
    export_hard_data: str = 'hard_mined_v1.npz'    # Optional path to export newly mined test failures as .npz (e.g. 'hard_mined_v1.npz')

