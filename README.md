OT-Guided Diffusion (MNIST) — Single Notebook

Everything (passive metrics + active training + evaluation) is in one notebook.

Quick start

Colab (recommended)

Open the notebook in Colab.

(Optional) drive.mount('/content/drive') to save outputs.

Run all cells top-to-bottom.



Local:
pip install torch torchvision
pip install git+https://github.com/lucidrains/denoising-diffusion-pytorch.git
pip install geomloss pot scipy scikit-learn wandb seaborn pandas tqdm

Open the notebook and run all cells.

What it does:

Passive phase: trains a tiny MNIST feature net; computes OT (Sinkhorn) and FID; stress tests by (a) adding noise and (b) removing classes.

Active phase: trains a baseline DDPM and an OT-enhanced DDPM (MSE + λ·OT in feature space).

Saves plots and checkpoints.

Main outputs (folder _ddpm_plots/)

three_panel_comparison.png — Real | Before | After.

before_after_ot_fid.png — OT & FID bars (baseline vs OT-enhanced).

epoch_compare_* — OT/FID vs epoch (before/after).

combined_ot.png, combined_fid.png — Passive stress tests (noise / missing classes).
(If not using Drive, change output paths to a local folder.)

Notes:

Lower epochs/batch size if you hit GPU limits.

You can rerun only the evaluation cells to regenerate plots from existing checkpoints.
