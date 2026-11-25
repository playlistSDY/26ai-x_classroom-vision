"""Read ordinary and HDR videos as correctly sized SDR BGR frames."""
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np


class VideoSource:
    def __init__(self, path, width=1280):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"영상을 찾을 수 없습니다: {self.path}")
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run(
            [executable, "-hide_banner", "-i", str(self.path), "-map", "0:v:0", "-an",
             "-frames:v", "0", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False,
        )
        if probe.returncode:
            raise RuntimeError(f"영상을 열지 못했습니다: {self.path.name}")
        self.hdr = any(name in probe.stderr for name in ("arib-std-b67", "smpte2084"))
        filters = []
        if self.hdr:
            filters.extend(("zscale=t=linear:npl=100", "format=gbrpf32le", "tonemap=tonemap=hable:desat=0",
                            "zscale=p=bt709:t=bt709:m=bt709:r=tv"))
        filters.append(f"scale=w='min({width},iw)':h=-2")
        self.reader = imageio_ffmpeg.read_frames(
            str(self.path), pix_fmt="bgr24", output_params=["-vf", ",".join(filters)],
        )
        self.metadata = next(self.reader)
        self.width, self.height = self.metadata["size"]
        self.fps = float(self.metadata["fps"] or 30)
        self.index = 0

    def read(self):
        try:
            data = next(self.reader)
        except StopIteration:
            return None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
        timestamp = self.index / self.fps
        self.index += 1
        return timestamp, frame

    def close(self):
        self.reader.close()
