import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smolagents import Tool

from tools.llm import ask_vision

VIDEO_DIR = Path(__file__).parent.parent / "cache" / "videos"
FRAMES_PER_GRID = 4
GRID_COLUMNS = 2
FRAME_WIDTH = 640
MAX_FRAMES = 120


def video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if not match:
        raise ValueError(f"Could not find a YouTube video id in {url!r}")
    return match.group(1)


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = (
        "Returns the captions/transcript of a YouTube video with timestamps. Cheap and fast: use it "
        "for questions about what is SAID in a video. It cannot see what is shown on screen."
    )
    inputs = {"url": {"type": "string", "description": "The YouTube video URL."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.proxies import GenericProxyConfig

        proxy = os.getenv("PROXY_URL")
        api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy) if proxy else None
        )
        try:
            transcript = api.fetch(video_id(url), languages=["en", "en-US", "en-GB"])
        except Exception as e:
            return f"No transcript available ({type(e).__name__}). Use youtube_video_analysis to look at the video instead."
        return "\n".join(f"[{s.start:.1f}s] {s.text}" for s in transcript)


class YouTubeVideoAnalysisTool(Tool):
    name = "youtube_video_analysis"
    description = (
        "Looks at what is SHOWN in a YouTube video: downloads it, samples frames at a fixed interval "
        "and asks a vision model about each frame. Returns one observation per timestamp. "
        "Use it for questions about visual content (objects, animals, counts, on-screen text). "
        "Expensive: first scan the whole video, then use start_seconds/end_seconds with a smaller "
        "interval to double-check the key moments before answering. Frames are analysed in "
        "separate batches, so pass what earlier calls found (e.g. identifications made in "
        "close-ups) as `context` to keep identifications consistent across frames."
    )
    inputs = {
        "url": {"type": "string", "description": "The YouTube video URL."},
        "question": {
            "type": "string",
            "description": "What to report for every frame, e.g. 'List each bird species visible and how many of each'.",
        },
        "seconds_between_frames": {
            "type": "number",
            "description": "Sampling interval in seconds (default 2). Smaller is more thorough but costs more.",
            "nullable": True,
        },
        "start_seconds": {"type": "number", "description": "Optional: only analyse from this time.", "nullable": True},
        "end_seconds": {"type": "number", "description": "Optional: only analyse up to this time.", "nullable": True},
        "context": {
            "type": "string",
            "description": "Optional: facts already established about the video, given to the vision model.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        url: str,
        question: str,
        seconds_between_frames: float | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        context: str | None = None,
    ) -> str:
        try:
            path = self._download(url)
        except Exception as e:
            return f"Could not download the video: {e}"

        frames = self._sample_frames(path, seconds_between_frames or 2.0, start_seconds or 0.0, end_seconds)
        if not frames:
            return "Could not read any frames from the video."
        grids = [frames[i:i + FRAMES_PER_GRID] for i in range(0, len(frames), FRAMES_PER_GRID)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(lambda g: self._ask_vision(g, question, context), grids))
        return f"Sampled {len(frames)} frames.\n" + "\n".join(reports)

    @staticmethod
    def _download(url: str) -> Path:
        import yt_dlp

        vid = video_id(url)
        existing = list(VIDEO_DIR.glob(f"{vid}.*"))
        if existing:
            return existing[0]
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        options = {
            # Video only: no audio needed, and low resolution keeps it small.
            "format": "bestvideo[height<=480][vcodec^=avc1]/bestvideo[height<=480]/best[height<=480]/best",
            "outtmpl": str(VIDEO_DIR / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "js_runtimes": {"node": {}},
        }
        if os.getenv("PROXY_URL"):
            options["proxy"] = os.getenv("PROXY_URL")
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
            return Path(ydl.prepare_filename(info))

    @staticmethod
    def _sample_frames(
        path: Path, interval: float, start: float = 0.0, end: float | None = None
    ) -> list[tuple[float, "np.ndarray"]]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        end = min(end or float("inf"), capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        interval = max(interval, (end - start) / MAX_FRAMES)
        frames, t = [], start
        while t < end:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if ok:
                height = int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
                frames.append((t, cv2.resize(frame, (FRAME_WIDTH, height))))
            t += interval
        capture.release()
        return frames

    @staticmethod
    def _ask_vision(grid: list, question: str, context: str | None = None) -> str:
        import cv2
        import numpy as np

        # Tile the frames into one image, each labelled with its timestamp.
        tiles = []
        for t, frame in grid:
            tile = frame.copy()
            cv2.rectangle(tile, (0, 0), (110, 32), (0, 0, 0), -1)
            cv2.putText(tile, f"t={t:.0f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(tile)
        while len(tiles) % GRID_COLUMNS:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + GRID_COLUMNS]) for i in range(0, len(tiles), GRID_COLUMNS)]
        _, jpeg = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

        prompt = (
            "This image is a grid of frames from one video; each frame is labelled with its timestamp "
            f"in the top-left corner. For EACH labelled frame answer: {question}\n"
            "Reply with one line per frame, formatted as 't=<seconds>s: <answer>'. Be precise and "
            "only report what is clearly visible. When identifying animals or objects, look closely at "
            "each individual: similar-looking ones may be different species/types (compare size, shape, "
            "colouring and markings), so do not assume they are all the same. Mention distinguishing "
            "features when you are unsure."
        )
        if context:
            prompt += (
                f"\nAlready established about this video (from other frames, e.g. close-ups): {context}\n"
                "Small or distant individuals may be the ones identified there - check each one against it."
            )
        return ask_vision(prompt, jpeg.tobytes())
