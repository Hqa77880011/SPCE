import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.signal import resample
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .signal import envelope, features, posterior, speed_metrics


VARIANTS = ["spce", "point", "mc", "oracle", "no_speed", "narrow", "shifted", "random", "raw_cnn"]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def generate(args):
    rng = np.random.default_rng(args.seed)
    records, speeds, labels, trials, profiles, orders = [], [], [], [], [], []
    t = np.arange(int(args.seconds * args.fs)) / args.fs
    fault_orders = [0., 4.95, 3.05, 2.3, .4]
    for profile in range(4):
        for trial in range(3):
            for label in range(5):
                speed = 15 + 3 * profile + 4 * np.sin(2 * np.pi * t / args.seconds + profile)
                phase = 2 * np.pi * np.cumsum(speed) / args.fs
                modulation = np.maximum(np.cos(fault_orders[label] * phase), 0) ** 12 if label else np.zeros_like(t)
                carrier = np.sin(2 * np.pi * .2 * args.fs * t)
                signal = .3 * np.sin(phase) + (.05 + modulation) * carrier + rng.normal(0, .05, len(t))
                records.append(signal)
                speeds.append(speed)
                labels.append(label)
                trials.append(trial)
                profiles.append(profile)
                orders.append(fault_orders[label])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, vibration=np.stack(records), speed_hz=np.stack(speeds),
        label=labels, recording=np.arange(len(records)), trial=trials, profile=profiles, fault_order=orders, fs=args.fs)


def import_manifest(args):
    records = []
    root = Path(args.input).resolve().parent
    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            raw = loadmat(path) if path.suffix.lower() == ".mat" else load(path)
            signal = np.asarray(raw[row["signal_key"]]).squeeze()
            fs = float(row["fs"])
            reference = np.asarray(raw[row["speed_key"]]).squeeze() if row.get("speed_key") else np.full(len(signal), np.nan)
            if row.get("encoder_key"):
                pulses = np.asarray(raw[row["encoder_key"]]).squeeze()
                level = .5 * (np.nanmin(pulses) + np.nanmax(pulses))
                edges = np.flatnonzero((pulses[1:] > level) & (pulses[:-1] <= level)) + 1
                cpr = float(row.get("encoder_cpr") or 1024)
                if len(edges) < 3:
                    raise ValueError("Encoder pulse stream has fewer than three rising edges")
                local_speed = fs / (np.diff(edges) * cpr)
                midpoints = .5 * (edges[1:] + edges[:-1])
                reference = np.interp(np.arange(len(signal)), midpoints, local_speed)
            if reference.shape != signal.shape:
                raise ValueError("speed_key must be an aligned speed-in-Hz array")
            records.append({"vibration": signal, "speed_hz": reference, "fs": fs,
                "label": int(row["label"]), "recording": int(row["recording"]),
                "trial": int(row["trial"]), "profile": int(row["profile"]),
                "fault_order": float(row.get("fault_order") or 0)})
    make_windows(records, args)


def prepare(args):
    raw = load(args.input)
    records = []
    fs = float(raw["fs"])
    for index, signal in enumerate(raw["vibration"]):
        records.append({"vibration": signal, "speed_hz": raw.get("speed_hz", np.full_like(raw["vibration"], np.nan))[index],
            "fs": fs, **{key: raw[key][index] for key in ["label", "recording", "trial", "profile", "fault_order"]}})
    make_windows(records, args)


def make_windows(records, args):
    rows = {key: [] for key in ["vibration", "amplitude", "speed_hz", "label", "recording", "trial", "profile", "fault_order"]}
    output_fs = args.sample_rate
    for record in records:
        fs = record["fs"]
        size, hop = int(args.window * fs), int(args.hop * fs)
        if size <= 0 or hop <= 0:
            raise ValueError("Window and hop must be positive")
        for start in range(0, len(record["vibration"]) - size + 1, hop):
            signal = record["vibration"][start:start + size]
            reference = record["speed_hz"][start:start + size]
            count = int(args.window * output_fs)
            rows["vibration"].append(resample(signal, count))
            rows["amplitude"].append(resample(envelope(signal, fs, args.envelope_band_hz), count))
            rows["speed_hz"].append(np.interp(np.linspace(0, len(reference) - 1, count), np.arange(len(reference)), reference))
            for key in ["label", "recording", "trial", "profile", "fault_order"]:
                rows[key].append(record[key])
    if not rows["vibration"]:
        raise ValueError("No complete windows in the supplied recordings")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in rows.items()}, fs=output_fs)


def split(data, axis, heldout):
    test = data[axis] == heldout
    remaining = np.unique(data[axis][~test])
    if len(remaining) < 2:
        raise ValueError("Need at least three trial/profile values for train/validation/test")
    validation = data[axis] == remaining[-1]
    train = ~(test | validation)
    for mask in [train, validation, test]:
        if not mask.any():
            raise ValueError("Each split must be nonempty")
    return train, validation, test


class RawCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, classes))

    def forward(self, signal):
        signal = signal / signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.layers(signal)


def extract(data, indices, config, variant, seed):
    vectors, details = [], []
    for index in indices:
        vector, detail = features(data["vibration"][index], float(data["fs"]), config, variant,
                                  seed + int(index), data["speed_hz"][index], data["amplitude"][index])
        vectors.append(vector)
        details.append(detail)
    return np.stack(vectors), details


def train(args):
    data = load(args.data)
    train_mask, val_mask, _ = split(data, args.axis, args.heldout)
    train_index, val_index = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    config = {"min_hz": args.min_hz, "max_hz": args.max_hz, "speed_bins": 64,
        "step_hz": args.step_hz, "trajectories": 7, "temperature": 1., "beta": 0.,
        "shift_hz": 2., "nominal_hz": 20., "orders": args.orders}
    start = time.perf_counter()
    best_nll = np.inf
    for temperature in [.5, 1., 2., 4.]:
        candidate = {**config, "temperature": temperature}
        values = []
        for index in train_index[:min(24, len(train_index))]:
            t, paths, weights = posterior(data["vibration"][index], float(data["fs"]), candidate, args.seed + int(index))
            metric = speed_metrics(t, paths, weights, data["speed_hz"][index], float(data["fs"]))
            if metric:
                values.append(metric["speed_nll"])
        if values and np.mean(values) < best_nll:
            config["temperature"], best_nll = temperature, float(np.mean(values))
    classes = np.unique(data["label"][train_mask])
    if not np.isin(data["label"][val_mask], classes).all():
        raise ValueError("Validation classes must be present in training")
    if args.variant == "raw_cnn":
        torch.manual_seed(args.seed)
        network = RawCNN(len(classes))
        optimizer = torch.optim.Adam(network.parameters(), lr=.001)
        signal = torch.from_numpy(data["vibration"][train_mask]).float()[:, None]
        labels = torch.tensor(np.searchsorted(classes, data["label"][train_mask]), dtype=torch.long)
        network.train()
        for _ in range(args.epochs):
            for ix in torch.randperm(len(signal)).split(32):
                loss = nn.functional.cross_entropy(network(signal[ix]), labels[ix])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        network.eval()
        with torch.no_grad():
            val_logits = network(torch.from_numpy(data["vibration"][val_mask]).float()[:, None]).numpy()
        classifier = {"weights": network.state_dict(), "classes": classes}
    else:
        training_features, _ = extract(data, train_index, config, args.variant, args.seed)
        validation_features, _ = extract(data, val_index, config, args.variant, args.seed)
        expanded = args.variant in {"spce", "narrow", "shifted", "random"}
        candidates = [0., .25, .5, 1.] if expanded else [0.]
        selected = None
        best = -np.inf
        for beta in candidates:
            train_features, val_features = training_features.copy(), validation_features.copy()
            if expanded:
                width = len(config["orders"])
                train_features[:, :width] -= beta * np.maximum(np.exp(train_features[:, width:]) - 1e-6, 0)
                val_features[:, :width] -= beta * np.maximum(np.exp(val_features[:, width:]) - 1e-6, 0)
            model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=args.epochs,
                random_state=args.seed)).fit(train_features, data["label"][train_mask])
            score = balanced_accuracy_score(data["label"][val_mask], model.predict(val_features))
            if score > best:
                selected, best, config["beta"] = model, score, beta
                val_logits = np.log(np.clip(model.predict_proba(val_features), 1e-12, 1))
        classifier = selected
    target = data["label"][val_mask]
    result = minimize_scalar(lambda log_t: log_loss(target, softmax(val_logits / np.exp(log_t), axis=1), labels=classes),
                             bounds=(-3, 3), method="bounded")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "classes": classes, "config": config,
        "temperature": float(np.exp(result.x)), "variant": args.variant, "axis": args.axis,
        "heldout": args.heldout, "seed": args.seed, "training_seconds": time.perf_counter() - start}, output / "model.joblib")


def evaluate(args):
    data, bundle = load(args.data), joblib.load(args.model)
    _, _, mask = split(data, bundle["axis"], bundle["heldout"])
    indices = np.flatnonzero(mask)
    variant, config = bundle["variant"], bundle["config"]
    start = time.perf_counter()
    if variant == "raw_cnn":
        network = RawCNN(len(bundle["classes"]))
        network.load_state_dict(bundle["classifier"]["weights"])
        network.eval()
        with torch.no_grad():
            logits = network(torch.from_numpy(data["vibration"][mask]).float()[:, None]).numpy()
        details = []
    else:
        vectors, details = extract(data, indices, config, variant, bundle["seed"])
        logits = np.log(np.clip(bundle["classifier"].predict_proba(vectors), 1e-12, 1))
    probability = softmax(logits / bundle["temperature"], axis=1)
    elapsed = time.perf_counter() - start
    recording_ids = np.unique(data["recording"][mask])
    predictions, labels, per_record = [], [], []
    for recording in recording_ids:
        selected = data["recording"][indices] == recording
        reference = np.unique(data["label"][indices[selected]])
        if len(reference) != 1:
            raise ValueError("Every recording must have a single health label")
        predictions.append(probability[selected].mean(0))
        labels.append(reference[0])
        per_record.append({"recording": int(recording), "label": int(reference[0]),
                           "probabilities": predictions[-1].tolist()})
    probabilities, labels = np.stack(predictions), np.asarray(labels)
    predicted = bundle["classes"][probabilities.argmax(1)]
    confidence, correct = probabilities.max(1), predicted == labels
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += np.mean(selected) * abs(confidence[selected].mean() - correct[selected].mean())
    metrics = {"variant": variant, "recordings": len(labels),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "diagnostic_ece": float(ece), "diagnostic_nll": float(log_loss(labels, probabilities, labels=bundle["classes"])),
        "training_seconds": bundle["training_seconds"], "evaluation_seconds": elapsed}
    chain = []
    for index, detail in zip(indices, details):
        measurements = speed_metrics(detail["time"], detail["paths"], detail["weights"],
                                     data["speed_hz"][index], float(data["fs"]))
        true_order = data["fault_order"][index]
        if true_order > 0:
            measurements["order_error"] = abs(config["orders"][int(np.argmax(detail["evidence"]))] - true_order)
            measurements["dispersion"] = float(np.mean(detail["dispersion"]))
        chain.append({"recording": int(data["recording"][index]), **measurements})
    for key in ["speed_rmse_hz", "speed_nll", "speed_coverage90", "order_error"]:
        recording_values = []
        for recording in recording_ids:
            values = [r[key] for r in chain if r["recording"] == recording and key in r]
            if values:
                recording_values.append(np.mean(values))
        if recording_values:
            metrics[key] = float(np.mean(recording_values))
    association = []
    for recording in recording_ids:
        rows = [r for r in chain if r["recording"] == recording and "order_error" in r]
        if rows:
            association.append([np.mean([r["order_error"] for r in rows]), np.mean([r["dispersion"] for r in rows])])
    if len(association) > 1 and np.all(np.std(association, axis=0) > 0):
        metrics["dispersion_order_error_correlation"] = float(np.corrcoef(np.asarray(association).T)[0, 1])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "metrics.json", metrics)
    save(output / "recordings.json", per_record)
    save(output / "measurement_chain.json", chain)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(Path(args.input).rglob("metrics.json"))
    if not paths:
        raise ValueError("No evaluated metrics found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ["balanced_accuracy", "macro_f1", "order_error", "speed_rmse_hz", "speed_nll", "speed_coverage90", "diagnostic_ece"]:
        values = [json.loads(p.read_text()).get(key) for p in paths]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([str(p.parent.relative_to(args.input)) for p in paths],
               [np.nan if v is None else v for v in values], color="#4c8198")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="spce")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--output", required=True)
    p.add_argument("--fs", type=float, default=4096)
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--seed", type=int, default=13)
    for name in ["prepare", "import-manifest"]:
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--window", type=float, default=1.)
        p.add_argument("--hop", type=float, default=.5)
        p.add_argument("--sample-rate", type=float, default=4096)
        p.add_argument("--envelope-band-hz", nargs=2, type=float)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="spce")
    p.add_argument("--axis", choices=["trial", "profile"], default="trial")
    p.add_argument("--heldout", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-hz", type=float, default=5.)
    p.add_argument("--max-hz", type=float, default=40.)
    p.add_argument("--step-hz", type=float, default=.8)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--orders", nargs="+", type=float, default=np.linspace(.2, 8, 96).tolist())
    p = sub.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
