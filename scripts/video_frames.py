"""
Make a video readable by an agent, by turning it into frames.

openclaw's model schema allows only text and image inputs -- declaring "video"
is rejected outright -- so no video reaches a model through the normal path,
whatever the provider supports. Sampling frames and handing those over as
images is the way around it, and it is what vision tooling generally does.

    video_frames.py clip.mp4                    ~12 frames spread across it
    video_frames.py clip.mp4 --fps 1            one frame per second
    video_frames.py clip.mp4 --max-frames 30
    video_frames.py clip.mp4 --start 30 --duration 15

Prints one frame path per line with its timestamp, so the caller can read the
interesting ones rather than all of them. Frames are downscaled by default:
a 4K still costs a lot of tokens and adds nothing over 768px for describing
what is happening.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Playwright also ships an ffmpeg, but its build decodes only mjpeg and libvpx
# -- it cannot open an h264 mp4, which is most real video. imageio-ffmpeg
# bundles a full static build (h264/hevc/vp9/av1), so prefer that.
PLAYWRIGHT_FFMPEG = Path.home() / ".cache/ms-playwright/ffmpeg-1011/ffmpeg-linux"


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    for cand in (shutil.which("ffmpeg"), str(PLAYWRIGHT_FFMPEG)):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(
        "no ffmpeg found. Install one with: pip install --user imageio-ffmpeg"
    )


def probe_duration(ffmpeg: str, path: Path) -> float:
    """Duration via ffmpeg itself -- ffprobe is not in the playwright bundle."""
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            stamp = line.split("Duration:")[1].split(",")[0].strip()
            try:
                h, m, s = stamp.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                break
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Extract frames from a video for an agent to read.")
    ap.add_argument("video")
    ap.add_argument("--outdir", default=None, help="default: <video>.frames/ next to the file")
    ap.add_argument("--fps", type=float, default=None, help="frames per second to sample")
    ap.add_argument("--max-frames", type=int, default=12,
                    help="cap; with no --fps, frames are spread evenly across the video")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds")
    ap.add_argument("--width", type=int, default=768, help="downscale width; 0 keeps original")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"no such video: {video}")

    ffmpeg = find_ffmpeg()
    outdir = Path(args.outdir).expanduser() if args.outdir else video.with_suffix(video.suffix + ".frames")
    if outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    total = probe_duration(ffmpeg, video)
    span = args.duration if args.duration is not None else max(total - args.start, 0.0)

    if args.fps:
        fps = args.fps
    elif span > 0:
        # Spread max_frames evenly rather than sampling a burst at the start.
        fps = args.max_frames / span
    else:
        fps = 1.0

    vf = f"fps={fps:.6f}"
    if args.width:
        vf += f",scale={args.width}:-2"

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", str(video)]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    cmd += ["-vf", vf, "-frames:v", str(args.max_frames), str(outdir / "frame-%04d.jpg")]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {(proc.stderr or '').strip()[:300]}")

    frames = sorted(outdir.glob("frame-*.jpg"))
    step = (1.0 / fps) if fps else 0.0
    rows = [{"path": str(f), "t": round(args.start + i * step, 2)} for i, f in enumerate(frames)]

    if args.json:
        print(json.dumps({"video": str(video), "duration_s": round(total, 2),
                          "frames": rows}, indent=2))
        return

    print(f"{len(rows)} frame(s) from {video.name} (duration {total:.1f}s) -> {outdir}")
    for r in rows:
        print(f"  t={r['t']:>7.2f}s  {r['path']}")
    if rows:
        print("\nRead these image files to see the video. Start with a few spread across "
              "the range; only read more if you need finer detail.")


if __name__ == "__main__":
    main()
