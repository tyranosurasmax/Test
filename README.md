# Test Repository

This repository contains experimental trading modules and standalone utilities.

## Instruments

### `oscillator3x.py`

Generates a WAV file from three configurable oscillators. Parameters can be
specified either on the command line or via a JSON configuration file.

Example:

```bash
python instruments/oscillator3x.py \
    --output demo.wav \
    --duration 2.0 \
    --osc1 "saw,semitone=0,amplitude=0.5" \
    --osc2 "sine,semitone=2,amplitude=0.3" \
    --osc3 "square,semitone=-2,amplitude=0.2"
```

A sample configuration is provided at `assets/oscillator3x_config.json` and can
be used with:

```bash
python instruments/oscillator3x.py --config assets/oscillator3x_config.json
```
