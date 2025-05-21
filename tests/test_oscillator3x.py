import os
import tempfile
import wave
import unittest

from instruments.oscillator3x import generate_3x_osc, DEFAULT_SAMPLE_RATE


class TestOscillator3x(unittest.TestCase):
    def test_generate_wav_duration_and_rate(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            filename = tf.name
        oscillators = [
            {"shape": "sine", "offset": 0.0, "amplitude": 0.5},
            {"shape": "square", "offset": 1.0, "amplitude": 0.3},
            {"shape": "saw", "offset": -1.0, "amplitude": 0.2},
        ]
        duration = 0.1
        generate_3x_osc(filename, duration, oscillators)
        try:
            with wave.open(filename, "rb") as wf:
                self.assertEqual(wf.getframerate(), DEFAULT_SAMPLE_RATE)
                frames = wf.getnframes()
                file_duration = frames / float(wf.getframerate())
                self.assertAlmostEqual(file_duration, duration, places=2)
        finally:
            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
