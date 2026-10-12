import re

import requests
from markdownify import markdownify
from smolagents import Tool

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "HF-Agents-Course-Final-Assignment/1.0 (educational project)"}
MAX_CHARS = 40_000


class WikipediaPageTool(Tool):
    name = "wikipedia_page"
    description = (
        "Fetches the full content of an English Wikipedia article as markdown (tables included). "
        "Optionally returns the article as it was at a given date, e.g. '2022-12-31' for the "
        "latest 2022 version. Use web_search first if you are unsure of the exact title."
    )
    inputs = {
        "title": {"type": "string", "description": "Article title, e.g. 'Mercedes Sosa'."},
        "as_of_date": {
            "type": "string",
            "description": "Optional date YYYY-MM-DD. Returns the last revision on or before that date.",
            "nullable": True,
        },
        "section": {
            "type": "string",
            "description": "Optional text to look for in section headings (e.g. 'Discography'); "
            "only matching sections are returned. Saves tokens on long articles.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, title: str, as_of_date: str | None = None, section: str | None = None) -> str:
        try:
            params = {"action": "parse", "format": "json", "prop": "text", "redirects": 1}
            if as_of_date:
                oldid = self._revision_at(title, as_of_date)
                if oldid is None:
                    return f"No revision of '{title}' found on or before {as_of_date}."
                params["oldid"] = oldid
            else:
                params["page"] = title
            data = requests.get(API_URL, params=params, headers=HEADERS, timeout=30).json()
        except Exception as e:
            return f"Request to Wikipedia failed: {e}"

        if "error" in data:
            return f"Wikipedia error: {data['error'].get('info')}. Try web_search to find the right title."

        markdown = markdownify(data["parse"]["text"]["*"], heading_style="ATX")
        markdown = re.sub(r"\[\[edit\]\([^)]*\)\]", "", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        if section:
            markdown = self._filter_sections(markdown, section)
        if len(markdown) > MAX_CHARS:
            markdown = markdown[:MAX_CHARS] + "\n\n...[truncated, use the `section` argument]"
        return f"# {data['parse']['title']}\n\n{markdown}"

    @staticmethod
    def _revision_at(title: str, date: str) -> int | None:
        params = {
            "action": "query", "format": "json", "prop": "revisions", "titles": title,
            "rvlimit": 1, "rvdir": "older", "rvstart": f"{date}T23:59:59Z", "rvprop": "ids",
            "redirects": 1,
        }
        pages = requests.get(API_URL, params=params, headers=HEADERS, timeout=30).json()["query"]["pages"]
        page = next(iter(pages.values()))
        revisions = page.get("revisions")
        return revisions[0]["revid"] if revisions else None

    @staticmethod
    def _filter_sections(markdown: str, keyword: str) -> str:
        parts = re.split(r"(?m)^(?=#{1,6} )", markdown)
        matches, keep_level = [], None
        for part in parts:
            heading = re.match(r"(#{1,6}) (.*)", part)
            if heading:
                level = len(heading.group(1))
                if keep_level is not None and level <= keep_level:
                    keep_level = None
                if keep_level is None and keyword.lower() in heading.group(2).lower():
                    keep_level = level
            if keep_level is not None:
                matches.append(part)
        return "".join(matches) if matches else f"No section matching '{keyword}'. Full article:\n\n{markdown}"
