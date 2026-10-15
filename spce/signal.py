import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, hilbert, resample, sosfiltfilt, stft
from scipy.special import logsumexp, softmax
from scipy.special import ndtr


def envelope(vibration, fs, band=None):
    band = [.1 * fs, .45 * fs] if band is None else band
    sos = butter(3, band, btype="bandpass", output="sos", fs=fs)
    return np.abs(hilbert(sosfiltfilt(sos, vibration)))


def posterior(vibration, fs, config, seed=0):
    target_fs = min(512., fs)
    count = max(128, int(len(vibration) / fs * target_fs))
    low_signal = resample(vibration, count)
    nperseg = min(256, count)
    freq, time, spectrum = stft(low_signal, target_fs, nperseg=nperseg,
                                noverlap=3 * nperseg // 4, boundary="zeros")
    speeds = np.linspace(config["min_hz"], config["max_hz"], config["speed_bins"])
    emission = np.zeros((len(time), len(speeds)))
    power = abs(spectrum) ** 2
    for index in range(len(time)):
        reference = np.median(power[:, index]) + 1e-12
        for harmonic in [1, 2, 3]:
            emission[index] += np.log1p(np.interp(speeds * harmonic, freq, power[:, index]) / reference) / harmonic
    emission /= config["temperature"]
    transition = -.5 * ((speeds[:, None] - speeds[None, :]) / config["step_hz"]) ** 2
    transition -= logsumexp(transition, axis=1, keepdims=True)
    forward = np.zeros_like(emission)
    forward[0] = emission[0] - np.log(len(speeds))
    forward[0] -= logsumexp(forward[0])
    for t in range(1, len(time)):
        forward[t] = emission[t] + logsumexp(forward[t - 1, :, None] + transition, axis=0)
        forward[t] -= logsumexp(forward[t])
    rng = np.random.default_rng(seed)
    trajectories = np.empty((config["trajectories"], len(time)))
    for k in range(len(trajectories)):
        state = rng.choice(len(speeds), p=softmax(forward[-1]))
        trajectories[k, -1] = speeds[state]
        for t in range(len(time) - 2, -1, -1):
            state = rng.choice(len(speeds), p=softmax(forward[t] + transition[:, state]))
            trajectories[k, t] = speeds[state]
    smooth = np.stack([np.convolve(np.pad(row, (1, 1), mode="edge"), [1/3, 1/3, 1/3], mode="valid")
                       for row in trajectories])
    return time, smooth, np.ones(len(trajectories)) / len(trajectories)


def cyclic_evidence(amplitude, fs, time, speed_hz, orders, samples_per_revolution=128):
    sample_time = np.arange(len(amplitude)) / fs
    speed = np.interp(sample_time, time, speed_hz)
    if not np.all(speed > 0):
        raise ValueError("Speed trajectories must remain positive")
    revolutions = cumulative_trapezoid(speed, sample_time, initial=0)
    angle = np.arange(0, revolutions[-1], 1 / samples_per_revolution)
    if len(angle) < 16:
        raise ValueError("Window has insufficient rotational extent for cyclic analysis")
    angular = np.interp(angle, revolutions, amplitude)
    angular = angular - angular.mean()
    taper = np.hanning(len(angular))
    spectrum = np.abs(np.fft.rfft(angular * taper)) ** 2
    bins = np.fft.rfftfreq(len(angular), 1 / samples_per_revolution)
    evidence = np.interp(orders, bins, spectrum, left=0, right=0)
    return np.log1p(evidence / (np.median(spectrum) + 1e-10))


def features(vibration, fs, config, variant, seed, encoder=None, amplitude=None):
    time, paths, weights = posterior(vibration, fs, config, seed)
    original_paths = paths.copy()
    point = np.average(paths, axis=0, weights=weights)
    if variant == "narrow":
        paths = point + .25 * (paths - point)
    elif variant == "shifted":
        paths = np.maximum(.1, paths + config["shift_hz"])
    elif variant == "random":
        rng = np.random.default_rng(seed + 100000)
        paths = rng.uniform(config["min_hz"], config["max_hz"], paths.shape)
        paths = np.stack([np.convolve(np.pad(row, (2, 2), mode="edge"), np.ones(5)/5, mode="valid") for row in paths])
    amplitude = envelope(vibration, fs) if amplitude is None else amplitude
    orders = np.asarray(config["orders"])
    if variant == "oracle":
        if encoder is None or not np.isfinite(encoder).all():
            raise ValueError("Encoder oracle requires a positive finite speed_hz reference")
        evidence = cyclic_evidence(amplitude, fs, np.arange(len(encoder)) / fs, encoder, orders)
        mean, dispersion = evidence, np.zeros_like(evidence)
        result = evidence
    elif variant == "no_speed":
        frequency = np.fft.rfftfreq(len(amplitude), 1 / fs)
        spectrum = abs(np.fft.rfft(amplitude - amplitude.mean())) ** 2
        result = np.log1p(np.interp(orders * config["nominal_hz"], frequency, spectrum) / (np.median(spectrum) + 1e-10))
        mean, dispersion = result, np.zeros_like(result)
    elif variant == "point":
        mean = cyclic_evidence(amplitude, fs, time, point, orders)
        dispersion = np.zeros_like(mean)
        result = mean
    else:
        evidence = np.stack([cyclic_evidence(amplitude, fs, time, path, orders) for path in paths])
        mean = weights @ evidence
        dispersion = np.sqrt(weights @ (evidence - mean) ** 2)
        result = mean if variant == "mc" else np.r_[mean - config["beta"] * dispersion,
                                                       np.log(dispersion + 1e-6)]
    effective_paths = original_paths if variant in {"point", "mc", "oracle", "no_speed"} else paths
    return result, {"time": time, "paths": effective_paths, "weights": weights,
                    "evidence": mean, "dispersion": dispersion}


def speed_metrics(time, paths, weights, reference, fs, bandwidth=.5):
    actual = np.interp(time, np.arange(len(reference)) / fs, reference)
    if not np.isfinite(actual).all() or np.any(actual <= 0):
        return {}
    mean = weights @ paths
    log_density = logsumexp(np.log(weights[:, None]) - .5 * ((paths - actual) / bandwidth) ** 2
                            - np.log(bandwidth * np.sqrt(2 * np.pi)), axis=0)
    def mixture_quantile(probability):
        low, high = paths.min(0) - 10 * bandwidth, paths.max(0) + 10 * bandwidth
        for _ in range(50):
            middle = .5 * (low + high)
            cumulative = weights @ ndtr((middle - paths) / bandwidth)
            low = np.where(cumulative < probability, middle, low)
            high = np.where(cumulative >= probability, middle, high)
        return .5 * (low + high)
    lower, upper = mixture_quantile(.05), mixture_quantile(.95)
    return {"speed_rmse_hz": float(np.sqrt(np.mean((mean - actual) ** 2))),
            "speed_nll": float(-log_density.mean()), "speed_coverage90": float(np.mean((actual >= lower) & (actual <= upper)))}
