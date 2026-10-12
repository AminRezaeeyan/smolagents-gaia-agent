from smolagents import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Searches the web with DuckDuckGo and returns the top results (title, URL, snippet). "
        "Tip: add 'site:en.wikipedia.org' to the query to restrict results to Wikipedia."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query."},
    }
    output_type = "string"

    def __init__(self, max_results: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_results = max_results

    def forward(self, query: str) -> str:
        from ddgs import DDGS

        try:
            results = DDGS().text(query, max_results=self.max_results)
        except Exception as e:
            return f"Search failed: {e}"
        if not results:
            return "No results found. Try a different or shorter query."
        listing = "\n\n".join(
            f"[{r.get('title', '')}]({r.get('href', '')})\n{r.get('body', '')}" for r in results
        )
        return listing + (
            "\n\nNOTE: snippets are partial and show the page as it is TODAY. Open the relevant page "
            "with visit_webpage or wikipedia_page before answering; if the question is about a past "
            "date, pass as_of_date to read the page as it was then."
        )
