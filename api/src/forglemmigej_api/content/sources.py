"""Download the raw learning-content sources (never committed; see README attribution)."""

import bz2
import shutil
import urllib.request
from pathlib import Path

TATOEBA = "https://downloads.tatoeba.org/exports/per_language"

# Local file name -> URL. ".bz2" URLs are decompressed while downloading.
SOURCES = {
    "da_50k.txt": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/da/da_50k.txt",
    "kaikki-danish.jsonl": "https://kaikki.org/dictionary/Danish/kaikki.org-dictionary-Danish.jsonl",
    "dan_sentences.tsv": f"{TATOEBA}/dan/dan_sentences.tsv.bz2",
    "dan-eng_links.tsv": f"{TATOEBA}/dan/dan-eng_links.tsv.bz2",
    "eng_sentences.tsv": f"{TATOEBA}/eng/eng_sentences.tsv.bz2",
}


def download_sources(raw_dir: Path, force: bool = False) -> list[str]:
    """Fetch missing source files into `raw_dir`. Returns the names downloaded."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, url in SOURCES.items():
        target = raw_dir / name
        if target.exists() and not force:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as out,
        ):
            source = bz2.BZ2File(response) if url.endswith(".bz2") else response
            shutil.copyfileobj(source, out, length=1 << 20)
        partial.replace(target)  # only complete files get the final name
        downloaded.append(name)
    return downloaded
