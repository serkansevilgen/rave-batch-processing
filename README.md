# Offline RAVE Batch Processing

This script processes long audio files with a TorchScript-exported RAVE model in Python/PyTorch, without using Max `nn~` and without realtime playback.

By default, it processes every `.wav` file in the input folder. You can optionally filter by suffix, for example:

- `*_S.wav`
- `*_T.wav`
- `*_N.wav`

## Files

- `offline_rave_batch.py`: batch processor
- `requirements.txt`: Python dependencies

## Install

From this project folder:

```bash
pip install -r requirements.txt
```

## Run

Use explicit input and output folders:

```bash
python offline_rave_batch.py \
  --model /path/to/rave-models/model.ts \
  --input-dir /path/to/input/audio \
  --output-dir /path/to/output/audio \
  --model-sr 48000
```

To process only the `S`, `T`, and `N` suffix files:

```bash
python offline_rave_batch.py \
  --model /path/to/rave-models/model.ts \
  --input-dir /path/to/input/audio \
  --output-dir /path/to/output/audio \
  --model-sr 48000 \
  --suffixes S,T,N
```

Existing output files are skipped. Add `--overwrite` to replace them:

```bash
python offline_rave_batch.py \
  --model /path/to/rave-models/model.ts \
  --input-dir /path/to/input/audio \
  --output-dir /path/to/output/audio \
  --model-sr 48000 \
  --overwrite
```

Check the matched files and output paths without loading the model or writing audio:

```bash
python offline_rave_batch.py \
  --model /path/to/rave-models/model.ts \
  --input-dir /path/to/input/audio \
  --output-dir /path/to/output/audio \
  --model-sr 48000 \
  --dry-run
```

## Useful Options

`--suffixes S,T,N`

Optional suffix filter. If omitted, all `.wav` files in the input folder are processed. If set to `S,T,N`, only `*_S.wav`, `*_T.wav`, and `*_N.wav` are processed.

`--chunk-seconds 30`

Controls how much audio is processed at once. Lower this if memory usage is too high.

`--overlap-seconds 0.25`

Adds overlap and crossfade between chunks to reduce clicks at chunk boundaries.

`--model-sr 48000`

Sets the RAVE model sample rate. Use this when the model sample rate differs from the input files. Your `guitar_iil_b2048_r48000_z16.ts` model appears to be a 48 kHz model, so use `--model-sr 48000`.

`--device auto`

Chooses `cuda`, then `mps`, then `cpu`. You can force one:

```bash
--device cpu
--device mps
--device cuda
```

`--stereo`

Keeps stereo input. By default, the script mixes files to mono, which is common for RAVE models.

`--dry-run`

Prints the matched input files and output paths without loading the RAVE model or writing audio.

## Output Names

For each input file, the script writes:

```text
<original_stem>_rave.wav
```

Example:

```text
2022-09-18-istanbul-bienali-hassas-sesler-sozlugu-soundwalk_S_rave.wav
```

## Notes

The script expects a TorchScript `.ts` RAVE export. This model exposes both `encode` and `decode`, so the default `--method auto` should use encode/decode processing.

Audio metadata and chunked reads use `soundfile`, not `torchaudio.info`, because some `torchaudio` builds do not expose the `info` helper.
