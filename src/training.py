import numpy as np
import torch
import torch.nn as nn

# return bacthed "normalized" targets numpy (stays with dense_loss)
def build_targets_batch(peaks_list, cfg, device='cpu'):

    grid_spacing = cfg.grid_spacing
    w_grid       = cfg.w_grid
    amp_range    = cfg.amp_range
    pos_range    = cfg.pos_range
    gamma_range  = cfg.gamma_range

    # build batched (normalized) target parameters into tensor form (nbatch, ngrid)
    B = len(peaks_list)  # number of samples (per batch)
    L = len(w_grid)  # number of grids per sample (w_grid: the starting points of each grid)
    # four parameters per grid
    presence_t = np.zeros((B, L), dtype=np.float32)
    offset_t = np.zeros((B, L), dtype=np.float32)
    amp_t = np.zeros((B, L), dtype=np.float32)
    gamma_t = np.zeros((B, L), dtype=np.float32)

    for b, peaks in enumerate(peaks_list):
        # each sample can and generally has multiple peaks -- (A, pos, gamma)
        for A, pos, gamma in peaks:
            # g is the index of the grid point in w_grid that is closest in value to pos (the location of the peak).
            # np.argmin(...) Finds the index (position) of the smallest value in that distance array
            g = int(np.argmin(np.abs(w_grid - pos))) # the closest grid to the peak location (pos) 
            presence_t[b, g] = 1.0
            offset_t[b, g] = (pos - w_grid[g]) / grid_spacing # g only gives the grid location need fine tune
            amp_t[b, g] = (A - amp_range[0]) / (amp_range[1] - amp_range[0])
            gamma_t[b, g] = (gamma - gamma_range[0]) / (gamma_range[1] - gamma_range[0])

    return presence_t, offset_t, amp_t, gamma_t


# --- Loss: standard BCE (binary cross) + masked regression ---
def dense_loss(preds, targets):
    # Unpack the tuples
    presence, offset, amplitude, gamma = preds
    presence_t, offset_t, amp_t, gamma_t = targets

    # binary for presnce (present 1 or not 0)
    presence_loss = nn.functional.binary_cross_entropy(presence, presence_t)
    # calculate only when peak is present
    mask = presence_t
    #PyTorch sums over all elements across all dimensions by default.
    reg_loss = torch.sum(mask*(offset-offset_t)**2
                       + mask*(amplitude-amp_t)**2
                       + mask*(gamma-gamma_t)**2) / (mask.sum()+1e-7)
    return presence_loss + reg_loss


def training_loop(n_epochs, train_loader, val_loader, model, opt, scheduler,
                  convergence_window=0, convergence_rel_tol=0.0,
                  convergence_min_epochs=0):
    """Train for up to n_epochs, restoring the best-val checkpoint on exit.

    Convergence early stop (disabled when convergence_rel_tol <= 0): once at least
    `convergence_min_epochs` have run, compare the mean val loss over the last
    `convergence_window` epochs to the mean over the `convergence_window` epochs
    before that; stop when their relative difference drops below
    `convergence_rel_tol`. The windowed means smooth per-epoch noise.
    """
    best_val, best_state = float("inf"), None
    val_hist = []

    # Track dataset sizes for accurate loss averaging
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(n_epochs):
        # ---------------- TRAINING PHASE ---------------- #
        model.train()
        epoch_loss = 0.0

        # DataLoader handles shuffling and yields batch variables directly
        # targes = (presence_t, offset_t, amp_t, gamma_t)
        for xb, *targets in train_loader:

            # reset
            opt.zero_grad()

            # Forward pass:  preds = (presence, offset, amplitude, gamma)
            preds = model(xb)

            # Calculate loss
            loss = dense_loss(preds, targets)

            loss.backward()
            opt.step()

            # Scale loss by batch size (xb.size(0) dynamically handles the last uneven batch)
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / n_train

        # ---------------- VALIDATION PHASE ---------------- #
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for xb, *targets in val_loader:

                preds = model(xb)

                loss_v = dense_loss(preds, targets)

                val_epoch_loss += loss_v.item() * xb.size(0)

        val_loss = val_epoch_loss / n_val
        val_hist.append(val_loss)

        # ---------------- SCHEDULING & LOGGING ---------------- #
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            # Use state_dict copy to avoid referencing memory that changes
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

        # ---------------- CONVERGENCE EARLY STOP ---------------- #
        w = convergence_window
        if (w and convergence_rel_tol > 0
                and epoch + 1 >= convergence_min_epochs
                and len(val_hist) >= 2 * w):
            prev = float(np.mean(val_hist[-2 * w:-w]))
            recent = float(np.mean(val_hist[-w:]))
            rel = abs(recent - prev) / (abs(prev) + 1e-8)
            if rel < convergence_rel_tol:
                print(f"Converged at epoch {epoch+1}: rel val change "
                      f"{rel:.2e} < {convergence_rel_tol:.0e} "
                      f"(mean of last {w} vs previous {w} epochs)")
                break

    # Load best weights back into model before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, best_state
