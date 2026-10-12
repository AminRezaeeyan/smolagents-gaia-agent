import re

import requests
from markdownify import markdownify
from smolagents import Tool

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) HF-Agents-Course-Final-Assignment/1.0"}
MAX_CHARS = 40_000


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Fetches a web page and returns its content as markdown. Use it to read a page found with "
        "web_search instead of relying on search snippets. With as_of_date it returns the archived "
        "version from the Wayback Machine closest to that date - use this when the question asks "
        "about how something was at a specific time (pages like rosters or rankings change)."
    )
    inputs = {
        "url": {"type": "string", "description": "The page URL."},
        "as_of_date": {
            "type": "string",
            "description": "Optional date YYYY-MM-DD: fetch the Wayback Machine snapshot closest to it.",
            "nullable": True,
        },
        "find": {
            "type": "string",
            "description": "Optional text to look for: only the parts of the page around it are returned.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, url: str, as_of_date: str | None = None, find: str | None = None) -> str:
        header = ""
        if as_of_date:
            snapshot = self._wayback_snapshot(url, as_of_date)
            if snapshot is None:
                return f"No Wayback Machine snapshot found for {url} near {as_of_date}."
            url, timestamp = snapshot
            header = f"(Archived snapshot from {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]})\n\n"
        try:
            response = requests.get(url, headers=HEADERS, timeout=40)
            response.raise_for_status()
        except Exception as e:
            return f"Could not fetch {url}: {e}"

        markdown = markdownify(response.text, heading_style="ATX", strip=["script", "style"])
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        if find:
            markdown = self._around(markdown, find)
        if len(markdown) > MAX_CHARS:
            markdown = markdown[:MAX_CHARS] + "\n\n...[truncated, use the `find` argument]"
        return header + markdown

    @staticmethod
    def _wayback_snapshot(url: str, date: str) -> tuple[str, str] | None:
        timestamp = date.replace("-", "")
        data = requests.get(
            "https://archive.org/wayback/available",
            params={"url": url, "timestamp": timestamp},
            headers=HEADERS,
            timeout=40,
        ).json()
        closest = data.get("archived_snapshots", {}).get("closest")
        if not closest or not closest.get("available"):
            return None
        # The "id_" flag returns the original page without the Wayback toolbar.
        raw_url = re.sub(r"/web/(\d+)/", r"/web/\1id_/", closest["url"])
        raw_url = raw_url.replace("http://", "https://", 1)  # plain http is refused with 403
        return raw_url, closest["timestamp"]

    @staticmethod
    def _around(markdown: str, text: str, context: int = 1500) -> str:
        positions = [m.start() for m in re.finditer(re.escape(text), markdown, flags=re.IGNORECASE)]
        if not positions:
            return f"'{text}' not found on the page. Full page:\n\n{markdown}"
        chunks, last_end = [], -1
        for pos in positions[:10]:
            start, end = max(pos - context, last_end), pos + context
            if start < end:
                chunks.append(markdown[start:end])
                last_end = end
        return "\n\n[...]\n\n".join(chunks)
