"""Fetch a few reference frames of NICE-SLAM's Replica sequences, without the 12 GB zip.

    python fetch_reference.py DEST [--scene room0] [--frames 0 500]

NICE-SLAM's Replica sequences were rendered by the original OpenGL renderer, so
they are a reference for this one (see tests/test_scene_data.py). The files are
read straight out of the remote zip with HTTP range requests (about 10 MB for
two frames). Writes DEST/Replica/<scene>/{traj.txt, results/...}.
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request
import zipfile

URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest")
    ap.add_argument("--scene", default="room0", help="NICE-SLAM name: room0..2, office0..4")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 500])
    args = ap.parse_args()

    base = f"Replica/{args.scene}"
    wanted = [f"{base}/traj.txt", "Replica/cam_params.json"]
    for i in args.frames:
        wanted += [f"{base}/results/frame{i:06d}.jpg", f"{base}/results/depth{i:06d}.png"]

    f = HttpRangeFile(URL)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    dest = os.path.realpath(args.dest)
    for name in wanted:
        target = os.path.realpath(os.path.join(dest, name))
        if not target.startswith(dest + os.sep):
            raise ValueError(f"unsafe path in zip: {name}")
        print(z.extract(name, dest))
    print(f"{f.fetched / 1e6:.1f} MB transferred")


if __name__ == "__main__":
    main()
