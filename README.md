# SPCE

Implementation of **Propagating Tacholess Speed Uncertainty into Cyclic Evidence for Variable-Speed Bearing Diagnosis**. SPCE transforms complete speed-trajectory hypotheses into angular-domain evidence before taking posterior moments. Its classifier receives dispersion-penalized mean evidence and log evidence dispersion.

## Install

Use Python 3.10 or later:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -e .
```

## Prepare recordings

The included signal generator supplies 60 recordings with five health states, four speed profiles, and three trials:

```bash
spce generate --output data/signals.npz --fs 4096 --seconds 6
spce prepare --input data/signals.npz --output data/windows.npz --window 1 --hop 0.5
```

The generated source defines a rotational component, a modulated resonance carrier, additive noise, and known instantaneous speed. It is a signal model, with its exact equations in `generate`. The fault-order conventions are 4.95, 3.05, 2.3, and 0.4 for its four faulty classes.

For the paper's real-data setting, obtain the [University of Ottawa bearing dataset, version 2](https://doi.org/10.17632/v43hmbwxpm.2) and prepare a CSV manifest. This repository does not bundle recordings. Each row describes one complete recording:

```csv
path,signal_key,speed_key,encoder_key,encoder_cpr,fs,label,recording,trial,profile,fault_order
records/run01.mat,vibration,speed_hz,,,200000,0,1,0,0,0
records/run02.mat,vibration,,encoder,1024,200000,1,2,0,0,4.95
```

Use the actual numeric variable names and bearing geometry of the downloaded files. `speed_key` denotes an aligned speed reference already in Hz. Alternatively, `encoder_key` denotes a pulse signal; rising edges and `encoder_cpr` define revolutions. Do not put raw encoder counts into `speed_key`. Both reference fields may be omitted for tacholess inference, in which case speed-reference metrics and the oracle require a separate reference export.

```bash
spce import-manifest --input data/manifest.csv --output data/windows.npz --window 1 --hop 0.5 --sample-rate 4096 --envelope-band-hz 5000 40000
```

Choose the envelope band from training/development data and the original acquisition bandwidth. It is applied before downsampling. The importer preserves two paths: downsampled vibration for low-frequency speed estimation and the raw CNN, and the original-rate Hilbert envelope resampled to the common processing grid. This preserves high-frequency resonance modulation when the original carrier lies above the processing Nyquist frequency.

The direct NPZ recording schema is `vibration[N,T]`, optional `speed_hz[N,T]`, `label[N]`, `recording[N]`, `trial[N]`, `profile[N]`, `fault_order[N]`, and scalar `fs`. `fault_order=0` excludes healthy recordings from order-error calculation. `prepare` generates fixed windows while retaining recording IDs and metadata. Neighboring windows never cross a fold.

## Train

```bash
spce train --data data/windows.npz --output runs/spce/trial0 --variant spce --axis trial --heldout 0 --epochs 100 --min-hz 5 --max-hz 40
spce evaluate --data data/windows.npz --model runs/spce/trial0/model.joblib --output runs/spce/trial0/evaluation
```

For a trial fold, the held-out trial is test, the largest remaining trial ID is validation, and all other trials are training. With three trials this trains on one, validates on one, and tests on one. Run `--heldout 0`, `1`, and `2` for all trial rotations. Use `--axis profile` and rotate all profile IDs for the speed-profile protocol. Preserve the original release's IDs if they differ from the example.

The speed estimator uses a harmonic STFT observation model and a positive-speed Markov trajectory prior. Forward filtering and backward sampling produce seven complete, smoothed trajectory hypotheses. The speed range is fixed by `--min-hz`/`--max-hz`; `--step-hz` controls temporal speed increments. Training encoder references select an emission temperature by speed-posterior NLL over a fixed candidate grid. Encoder samples are not used by the tacholess test feature path.

Each trajectory is integrated in revolutions. The fixed envelope is interpolated on a uniform angular grid at 128 samples/revolution. A tapered angular spectrum supplies evidence at the shared candidate orders. The default order grid spans 0.2–8; use `--orders` to supply the candidate set fixed by the actual bearing geometry and harmonics.

SPCE computes posterior mean and standard deviation of evidence, then forms `[mean-beta*dispersion, log(dispersion+1e-6)]`. Validation balanced accuracy selects `beta` from 0, 0.25, 0.5, and 1. The classifier is a standardized one-hidden-layer width-64 MLP. A scalar temperature fitted on validation probabilities calibrates diagnostic confidence. `model.joblib` stores the classifier, fixed signal configuration, fold, seeds, and diagnostic temperature.

## Baselines and posterior interventions

```bash
spce train --data data/windows.npz --output runs/point/trial0 --variant point --axis trial --heldout 0
spce evaluate --data data/windows.npz --model runs/point/trial0/model.joblib --output runs/point/trial0/evaluation
spce train --data data/windows.npz --output runs/mc/trial0 --variant mc --axis trial --heldout 0
spce evaluate --data data/windows.npz --model runs/mc/trial0/model.joblib --output runs/mc/trial0/evaluation
spce plot --input runs --output runs/figures
```

| Variant | Feature construction |
| --- | --- |
| `spce` | Dispersion-penalized mean and log dispersion after trajectory-wise transformation |
| `point` | Average speed first, then integrate/resample and extract evidence |
| `mc` | Transform all seven trajectories, then retain mean evidence only |
| `oracle` | Use encoder speed in the same cyclic operator |
| `no_speed` | Envelope spectrum on a fixed nominal-speed frequency/order correspondence |
| `narrow` | Contract trajectories around their mean by 0.25 before propagation |
| `shifted` | Add 2 Hz to supported trajectories before propagation |
| `random` | Replace hypotheses with smoothed random positive speed trajectories |
| `raw_cnn` | Two convolution layers and global pooling on raw vibration windows |

Every tacholess variant starts from the same trajectory sampler, seeds, speed calibration, windows, angular grid, and evidence operator. Expanded representations use the same validation beta grid. Point and MC features have one channel per order; SPCE has two. The raw CNN has 16/32 convolution channels and uses cross-entropy with Adam at 0.001. `--epochs` controls classifier training for every variant.

## Metrics and output interpretation

Window probabilities are averaged within recordings before computing diagnostic metrics. The independent unit remains the recording.

| File/metric | Meaning |
| --- | --- |
| `metrics.json` | Recording balanced accuracy, macro-F1, diagnostic ECE/NLL, order error, speed RMSE/NLL/coverage, correlation, and runtime |
| `recordings.json` | Recording IDs, labels, and aggregated health-state probabilities |
| `measurement_chain.json` | Window-level speed-reference and derived-order diagnostics with recording IDs |
| `speed_rmse_hz` | Error of the posterior mean against the aligned encoder-speed reference |
| `speed_nll` | Mean pointwise Gaussian-mixture NLL with 0.5 Hz kernel bandwidth |
| `speed_coverage90` | Coverage of exact 5%/95% quantiles of that Gaussian mixture |
| `order_error` | Absolute difference between the strongest candidate order and the declared fault order |
| `diagnostic_ece` | Fifteen-bin expected calibration error after recording aggregation |
| `dispersion_order_error_correlation` | Recording-level correlation of evidence dispersion with realized order error |

Speed and order diagnostics are averaged within each recording and then across recordings. Healthy recordings are excluded from order error. Missing speed references omit the corresponding speed metrics; they are not replaced with estimated-speed targets. Zero-variance or insufficient correlation data omit the correlation. Charts use only metrics produced by evaluated runs.

## Implementation conventions

The manuscript specifies a trajectory posterior interface rather than a complete tacholess estimator. This implementation chooses the harmonic Markov estimator above and represents its posterior with seven equally weighted samples. The Gaussian kernel is an explicit continuous-density convention for evaluating a finite trajectory mixture; its calibration is separate from diagnostic temperature scaling.

Envelope filtering precedes angular resampling. Default filtering uses 0.10–0.45 of the original sample rate when no absolute band is supplied. Candidate orders, speed limits, and envelope bands must reflect the chosen bearing and acquisition chain. The no-speed baseline uses a nominal 20 Hz correspondence. These fixed conventions are defined in the saved configuration and signal-processing source.
