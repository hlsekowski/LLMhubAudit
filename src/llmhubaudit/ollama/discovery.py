"""Model discovery for the Ollama audit module.

Handles fetching candidate models from the Ollama registry, filtering by
tool support and parameter size, and extracting parameter sizes from model
names.
"""

from __future__ import annotations

import re

import requests  # type: ignore[import-untyped]


def _param_size_b(model_name: str) -> float | None:
    """Extract parameter size in billions from a model name/tag, or None."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if not matches:
        return None
    return max(float(m) for m in matches)


def _scrape_links(url: str) -> list[str]:
    """Return sorted unique hrefs matching /library/... from an Ollama page."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"registry scrape {url}: {e}")
        return []
    return sorted(set(re.findall(r'href="(/library/[^"]+)"', r.text)))


def _repo_has_tools(repo: str) -> bool:
    """Check the Ollama library page for a 'tools' capability badge."""
    try:
        r = requests.get(f"https://ollama.com/library/{repo}", timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"registry scrape {repo}: {e}")
        return False
    badges = re.findall(
        r'<span class="[^"]*">\s*(tools|vision|thinking|completion|embedding|audio|insert)\s*</span>',
        r.text,
    )
    return "tools" in badges


def fetch_registry_models(
    limit: int = 200,
    offset: int = 0,
    min_params: float | None = None,
    max_params: float | None = None,
) -> list[str]:
    """
    Fetch candidate model names from the Ollama library (scraped HTML).

    Filters to models with parameter size between ``min_params`` and
    ``max_params`` (billions) and that advertise a ``tools`` capability
    badge on their library page, so models without tool support are never
    pulled. ``offset`` skips the first ``offset`` matching models
    (internal paging).

    Returns a list of ``"name:tag"`` strings.
    """
    repos: list[str] = []
    for href in _scrape_links("https://ollama.com/library"):
        repo = href[len("/library/") :]
        if ":" in repo:
            continue
        repos.append(repo)

    models: list[str] = []
    skipped = 0
    for repo in repos:
        if not _repo_has_tools(repo):
            print(f"skipping {repo}: no 'tools' badge on library page")
            continue
        for href in _scrape_links(f"https://ollama.com/library/{repo}/tags"):
            tag = href[len(f"/library/{repo}:") :]
            full = f"{repo}:{tag}"
            ps = _param_size_b(full)
            if ps is None:
                continue
            if min_params is not None and ps < min_params:
                continue
            if max_params is not None and ps > max_params:
                continue
            if skipped < offset:
                skipped += 1
                continue
            models.append(full)
            if len(models) >= limit:
                break
        if len(models) >= limit:
            break
    return models


def load_candidates(
    from_registry: bool = False,
    registry_limit: int = 200,
    offset: int = 0,
    min_params: float | None = None,
    max_params: float | None = None,
) -> list[str]:
    """Load candidate model names, optionally from the Ollama registry."""
    if from_registry:
        return fetch_registry_models(
            limit=registry_limit,
            offset=offset,
            min_params=min_params,
            max_params=max_params,
        )
    return []
