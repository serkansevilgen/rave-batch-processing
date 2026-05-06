#!/usr/bin/env python3
"""Offline batch processing for TorchScript-exported RAVE models.

Example:
    python offline_rave_batch.py \
        --model /path/to/rave_model.ts \
        --input-dir /path/to/input_audio \
        --output-dir /path/to/output_audio
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized since it had shape.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process WAV files with RAVE offline."
    )
    parser.add_argument("--model", required=True, type=Path, help="TorchScript RAVE .ts model")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing input WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where processed WAV files will be written.",
    )
    parser.add_argument(
        "--suffixes",
        default=None,
        help="Optional comma-separated postfixes to process, e.g. S,T,N for *_S.wav files. Default: all .wav files.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    parser.add_argument(
        "--model-sr",
        type=int,
        default=None,
        help="Model sample rate. If omitted, files are processed at their original sample rate.",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Keep stereo input. Default mixes files to mono, which is typical for RAVE models.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Processing device. Default chooses cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=("auto", "encode_decode", "forward"),
        help="Use encode/decode methods or the model forward call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching files and output paths without loading the model or writing audio.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def target_files(input_dir: Path, suffixes: Iterable[str] | None) -> list[Path]:
    if suffixes is None:
        return sorted(input_dir.glob("*.wav"))

    files: list[Path] = []
    for suffix in suffixes:
        files.extend(sorted(input_dir.glob(f"*_{suffix}.wav")))
    return sorted(set(files))


def run_rave(model: torch.jit.ScriptModule, audio: torch.Tensor, method: str) -> torch.Tensor:
    """Run one [batch, channels, samples] chunk through a RAVE export."""
    with torch.inference_mode():
        if method in ("auto", "encode_decode") and hasattr(model, "encode") and hasattr(model, "decode"):
            return model.decode(model.encode(audio))
        if method == "encode_decode":
            raise RuntimeError("Model does not expose encode/decode methods.")

        out = model(audio)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def read_audio_chunk(path: Path, frame_offset: int, num_frames: int) -> torch.Tensor:
    audio, _ = sf.read(
        path,
        start=frame_offset,
        frames=num_frames,
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(audio).transpose(0, 1).contiguous()


def process_file(
    path: Path,
    output_path: Path,
    model: torch.jit.ScriptModule,
    device: torch.device,
    chunk_seconds: float,
    overlap_seconds: float,
    model_sr: int | None,
    stereo: bool,
    method: str,
) -> None:
    info = sf.info(path)
    source_sr = info.samplerate
    processing_sr = model_sr or source_sr
    chunk_frames = max(1, int(chunk_seconds * source_sr))
    overlap_frames = max(0, int(overlap_seconds * source_sr))
    overlap_frames = min(overlap_frames, chunk_frames // 2)
    step_frames = chunk_frames - overlap_frames

    resample_in = None
    resample_out = None
    if source_sr != processing_sr:
        resample_in = torchaudio.transforms.Resample(source_sr, processing_sr).to(device)
        resample_out = torchaudio.transforms.Resample(processing_sr, source_sr).to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous_tail: torch.Tensor | None = None
    total_frames = info.frames

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=source_sr,
        channels=2 if stereo and info.channels > 1 else 1,
        subtype="PCM_24",
    ) as writer:
        for frame_offset in tqdm(
            range(0, total_frames, step_frames),
            desc=path.name,
            leave=False,
        ):
            audio = read_audio_chunk(path, frame_offset, chunk_frames)
            if not stereo:
                audio = audio.mean(dim=0, keepdim=True)

            input_len = audio.shape[-1]
            audio = audio.unsqueeze(0).to(device)
            if resample_in is not None:
                audio = resample_in(audio)

            processed = run_rave(model, audio, method)
            if resample_out is not None:
                processed = resample_out(processed)

            processed = fit_length(processed, input_len)
            chunk = processed.squeeze(0).detach().cpu()

            if previous_tail is not None and overlap_frames > 0:
                fade_len = min(overlap_frames, previous_tail.shape[-1], chunk.shape[-1])
                fade_out = torch.linspace(1.0, 0.0, fade_len).unsqueeze(0)
                fade_in = torch.linspace(0.0, 1.0, fade_len).unsqueeze(0)
                blended = previous_tail[:, -fade_len:] * fade_out + chunk[:, :fade_len] * fade_in
                writer.write(blended.transpose(0, 1).numpy())
                write_start = fade_len
            else:
                write_start = 0

            if overlap_frames > 0 and chunk.shape[-1] > overlap_frames:
                body = chunk[:, write_start:-overlap_frames]
                previous_tail = chunk[:, -overlap_frames:]
            else:
                body = chunk[:, write_start:]
                previous_tail = None

            if body.numel() > 0:
                writer.write(body.transpose(0, 1).numpy())

        if previous_tail is not None and previous_tail.numel() > 0:
            writer.write(previous_tail.transpose(0, 1).numpy())


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = args.output_dir
    suffixes = None
    if args.suffixes:
        suffixes = [suffix.strip() for suffix in args.suffixes.split(",") if suffix.strip()]

    files = target_files(args.input_dir, suffixes)
    if not files:
        raise SystemExit(f"No matching files found in {args.input_dir}")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(files)}")

    if args.dry_run:
        print("Dry run: no model loaded and no audio written.")
        for path in files:
            output_path = output_dir / f"{path.stem}_rave.wav"
            status = "exists" if output_path.exists() else "new"
            action = "overwrite" if output_path.exists() and args.overwrite else status
            print(f"{path.name} -> {output_path} [{action}]")
        return

    model = torch.jit.load(str(args.model), map_location=device).eval()

    for path in files:
        output_path = output_dir / f"{path.stem}_rave.wav"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue
        process_file(
            path=path,
            output_path=output_path,
            model=model,
            device=device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            model_sr=args.model_sr,
            stereo=args.stereo,
            method=args.method,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
