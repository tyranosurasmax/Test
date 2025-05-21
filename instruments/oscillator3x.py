import numpy as np
import wave
import struct
from typing import List, Dict

DEFAULT_SAMPLE_RATE = 44100


def _waveform(
    shape: str, frequency: float, duration: float, amplitude: float, sample_rate: int
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
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


def generate_3x_osc(
    filename: str,
    duration: float,
    oscillators: List[Dict],
    base_frequency: float = 440.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    if len(oscillators) != 3:
        raise ValueError("Three oscillator definitions required")

    combined = np.zeros(int(sample_rate * duration))
    for osc in oscillators:
        shape = osc.get("shape", "sine")
        offset = osc.get("offset", 0.0)
        amp = osc.get("amplitude", 1.0)
        freq = base_frequency + offset
        combined += _waveform(shape, freq, duration, amp, sample_rate)

    max_amp = np.max(np.abs(combined))
    if max_amp > 0:
        combined = combined / max_amp

    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for sample in combined:
            wf.writeframes(struct.pack("<h", int(sample * 32767)))


if __name__ == "__main__":
    osc_specs = [
        {"shape": "saw", "offset": 0.0, "amplitude": 0.5},
        {"shape": "sine", "offset": 2.0, "amplitude": 0.3},
        {"shape": "square", "offset": -2.0, "amplitude": 0.2},
    ]
    generate_3x_osc("output.wav", 3.0, osc_specs)
