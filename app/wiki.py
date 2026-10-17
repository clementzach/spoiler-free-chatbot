"""Wikipedia lookup tools for the chat model: search, then fetch an article."""
import html
import json
import re

import httpx

from .resilience import with_retries

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "spoiler-free-chatbot/0.1 (https://github.com/clementzach/spoiler-free-chatbot)"}
MAX_CHARS = 6000
RETRIES = 2
LOOKUP_FAILED = "Lookup failed"  # prefix of the text a failed tool call returns to the model

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search English Wikipedia. Returns up to 5 article titles with snippets. "
            "Use this first; you don't need to know the exact title.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_page",
            "description": "Fetch a Wikipedia article as plain text. By default only the introduction "
            "(usually spoiler-light). Set full=true for the whole article, which often contains "
            "detailed plot sections.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "full": {"type": "boolean"}},
                "required": ["title"],
            },
        },
    },
]


async def _get(http: httpx.AsyncClient, **params) -> dict:
    async def call() -> dict:
        r = await http.get(
            API, params={"format": "json", "formatversion": 2, **params}, headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        return r.json()

    return await with_retries(call, RETRIES, label="wikipedia")


async def search(http: httpx.AsyncClient, query: str) -> str:
    data = await _get(http, action="query", list="search", srsearch=query, srlimit=5)
    hits = data["query"]["search"]
    if not hits:
        return "No results."
    return "\n".join(
        f"{h['title']}: {html.unescape(re.sub(r'<[^>]+>', '', h['snippet']))}" for h in hits
    )


async def page(http: httpx.AsyncClient, title: str, full: bool = False) -> str:
    params = dict(action="query", prop="extracts", explaintext=1, titles=title, redirects=1)
    if not full:
        params["exintro"] = 1
    pages = (await _get(http, **params))["query"]["pages"]
    p = pages[0]
    if p.get("missing") or not p.get("extract"):
        return f"No article titled {title!r}. Try wikipedia_search."
    text = p["extract"]
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n[truncated]"
    return f"{p['title']}\n\n{text}"


async def run_tool(http: httpx.AsyncClient, name: str, arguments: str) -> str:
    """Never raises: errors go back to the model as text so it can carry on without them."""
    try:
        args = json.loads(arguments or "{}")
        if name == "wikipedia_search":
            return await search(http, args["query"])
        if name == "wikipedia_page":
            return await page(http, args["title"], bool(args.get("full")))
        return f"Unknown tool {name!r}."
    except Exception as exc:
        return f"{LOOKUP_FAILED}: {type(exc).__name__}"
