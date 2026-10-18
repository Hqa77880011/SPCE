# SPCE

Implementation of **Propagating Tacholess Speed Uncertainty into Cyclic Evidence for Variable-Speed Bearing Diagnosis**.

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Generate modeled bearing signals with five health states, four speed profiles and three trials. The source combines rotational components, a modulated resonance and additive noise; it is not a measured bearing dataset. NPZ input uses `vibration[N,T]`, optional `speed_hz[N,T]`, `label`, `recording`, `trial`, `profile`, `fault_order` and scalar `fs`. Windows from a recording stay in one fold.

```bash
spce generate --output data/signals.npz --fs 4096 --seconds 6 --seed 13
spce prepare --input data/signals.npz --output data/windows.npz --window 1 --hop 0.5
```

For real recordings, obtain the [University of Ottawa bearing data v2](https://doi.org/10.17632/v43hmbwxpm.2). A CSV manifest uses `path,signal_key,speed_key,encoder_key,encoder_cpr,fs,label,recording,trial,profile,fault_order`. Paths and actual MAT keys identify the source vectors. `speed_key` is speed in Hz; encoder pulses instead use `encoder_key` and pulses/revolution. Healthy records use fault order zero.

```bash
spce import-manifest --input data/manifest.csv --output data/real_windows.npz --window 1 --hop 0.5 --sample-rate 4096 --envelope-band-hz 5000 40000
```

Choose the original-rate envelope band and speed limits from acquisition metadata and development data.

## Training and inference

```bash
spce train --data data/windows.npz --output runs/full --variant spce --axis trial --heldout 0 --epochs 100 --min-hz 5 --max-hz 40 --seed 13
```

## Evaluation

```bash
spce evaluate --data data/windows.npz --model runs/full/model.joblib --output runs/full/evaluation
```

## Ablations

Variants: `spce`, `point`, `mc`, `oracle`, `no_speed`, `narrow`, `shifted`, `random`, `raw_cnn`. Reuse the same fold and seed. Rotate `--heldout 0/1/2` for the generated trial folds; `--axis profile` holds out a speed profile.

```bash
spce train --data data/windows.npz --output runs/point --variant point --axis trial --heldout 0 --epochs 100 --min-hz 5 --max-hz 40 --seed 13
spce evaluate --data data/windows.npz --model runs/point/model.joblib --output runs/point/evaluation
```

## Outputs

`model.joblib` stores classifier, signal configuration, fold and seeds. Evaluation writes `metrics.json`, recording-level probabilities in `recordings.json`, and speed/order diagnostics in `measurement_chain.json`. Missing speed references omit reference-dependent metrics.

```bash
spce plot --input runs --output runs/figures
```
