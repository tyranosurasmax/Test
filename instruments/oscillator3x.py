import argparse
import json
import struct
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

DEFAULT_SAMPLE_RATE = 44100


def _waveform(
    shape: str,
    frequency: float,
    duration: float,
    amplitude: float,
    sample_rate: int,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    t = (t + phase) % duration
    if shape == "sine":
        signal = np.sin(2 * np.pi * frequency * t)
    elif shape == "square":
        signal = np.sign(np.sin(2 * np.pi * frequency * t))
    elif shape == "saw":
        signal = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    elif shape == "triangle":
        signal = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
    elif shape == "noise":
        signal = np.random.uniform(-1, 1, t.shape)
    else:
        raise ValueError(f"Unsupported wave shape: {shape}")
    return amplitude * signal


def _adsr_envelope(
    total_samples: int,
    sample_rate: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    a = int(sample_rate * attack)
    d = int(sample_rate * decay)
    r = int(sample_rate * release)
    s = max(total_samples - a - d - r, 0)
    attack_env = np.linspace(0, 1, a, False)
    decay_env = np.linspace(1, sustain, d, False)
    sustain_env = np.full(s, sustain)
    release_env = np.linspace(sustain, 0, r, False)
    env = np.concatenate([attack_env, decay_env, sustain_env, release_env])
    if len(env) < total_samples:
        env = np.pad(env, (0, total_samples - len(env)), constant_values=sustain)
    return env[:total_samples]


@dataclass
class OscSpec:
    shape: str = "sine"
    semitone: float = 0.0
    cents: float = 0.0
    amplitude: float = 1.0
    phase: float = 0.0

    def frequency(self, base_freq: float) -> float:
        factor = 2 ** (self.semitone / 12.0 + self.cents / 1200.0)
        return base_freq * factor


@dataclass
class InstrumentConfig:
    output: str = "output.wav"
    duration: float = 3.0
    base_frequency: float = 440.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    oscillators: List[OscSpec] = None
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.8
    release: float = 0.1

    @staticmethod
    def from_dict(data: Dict) -> "InstrumentConfig":
        osc_specs = [
            OscSpec(**osc) for osc in data.get("oscillators", [])
        ]
        return InstrumentConfig(
            output=data.get("output", "output.wav"),
            duration=data.get("duration", 3.0),
            base_frequency=data.get("base_frequency", 440.0),
            sample_rate=data.get("sample_rate", DEFAULT_SAMPLE_RATE),
            oscillators=osc_specs,
            attack=data.get("attack", 0.01),
            decay=data.get("decay", 0.1),
            sustain=data.get("sustain", 0.8),
            release=data.get("release", 0.1),
        )


def generate_3x_osc(cfg: InstrumentConfig) -> None:
    if not cfg.oscillators or len(cfg.oscillators) != 3:
        raise ValueError("Three oscillator definitions required")

    total_samples = int(cfg.sample_rate * cfg.duration)
    t = np.linspace(0, cfg.duration, total_samples, endpoint=False)
    combined = np.zeros_like(t)

    for osc in cfg.oscillators:
        freq = osc.frequency(cfg.base_frequency)
        combined += _waveform(
            osc.shape, freq, cfg.duration, osc.amplitude, cfg.sample_rate, osc.phase
        )

    env = _adsr_envelope(
        total_samples,
        cfg.sample_rate,
        cfg.attack,
        cfg.decay,
        cfg.sustain,
        cfg.release,
    )
    combined *= env
    max_amp = np.max(np.abs(combined))
    if max_amp > 0:
        combined /= max_amp

    with wave.open(cfg.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sample_rate)
        for sample in combined:
            wf.writeframes(struct.pack("<h", int(sample * 32767)))


def _parse_osc(text: str) -> OscSpec:
    parts = text.split(",")
    kwargs: Dict[str, float] = {}
    shape = parts[0]
    for part in parts[1:]:
        key, _, value = part.partition("=")
        kwargs[key] = float(value)
    return OscSpec(shape=shape, **kwargs)


def _load_config(path: Optional[str], args: argparse.Namespace) -> InstrumentConfig:
    if path:
        with open(path, "r") as f:
            data = json.load(f)
        return InstrumentConfig.from_dict(data)

    osc_texts = [args.osc1, args.osc2, args.osc3]
    oscillators = [_parse_osc(t) for t in osc_texts if t]
    cfg = InstrumentConfig(
        output=args.output,
        duration=args.duration,
        base_frequency=args.frequency,
        sample_rate=args.sample_rate,
        oscillators=oscillators,
        attack=args.attack,
        decay=args.decay,
        sustain=args.sustain,
        release=args.release,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="3x oscillator generator")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=440.0)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--osc1", default="saw,semitone=0,amplitude=0.5")
    parser.add_argument("--osc2", default="sine,semitone=2,amplitude=0.3")
    parser.add_argument("--osc3", default="square,semitone=-2,amplitude=0.2")
    parser.add_argument("--attack", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--sustain", type=float, default=0.8)
    parser.add_argument("--release", type=float, default=0.1)
    parser.add_argument("--config", help="JSON configuration file")

    args = parser.parse_args()
    cfg = _load_config(args.config, args)
    generate_3x_osc(cfg)


if __name__ == "__main__":
    main()
