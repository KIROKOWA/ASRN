# Security

ASRN is research code. If you find a security problem, please open an issue, or, for something that should not be
public, use the contact details on the author's GitHub profile.

**Loading checkpoints.** `asrn.load` / `asrn.cargar` read files with `torch.load(..., weights_only=True)`: a checkpoint
cannot run code when it is loaded. Older checkpoints that need full unpickling can be opened only with the explicit
opt-in `confiar=True`; use it only for files you trust.
