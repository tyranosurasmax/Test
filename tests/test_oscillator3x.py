import os
import tempfile
import unittest
import wave

from instruments.oscillator3x import (
    DEFAULT_SAMPLE_RATE,
    InstrumentConfig,
    OscSpec,
    generate_3x_osc,
)


class Oscillator3xTests(unittest.TestCase):
    def test_generate_wav_length(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cfg = InstrumentConfig(
            output=path,
            duration=1.0,
            base_frequency=440.0,
            sample_rate=DEFAULT_SAMPLE_RATE,
            oscillators=[
                OscSpec(shape="sine"),
                OscSpec(shape="sine", semitone=12, amplitude=0.5),
                OscSpec(shape="square", semitone=-12, amplitude=0.5),
            ],
        )
        generate_3x_osc(cfg)
        with wave.open(path) as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), DEFAULT_SAMPLE_RATE)
            self.assertEqual(wf.getnframes(), DEFAULT_SAMPLE_RATE)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
