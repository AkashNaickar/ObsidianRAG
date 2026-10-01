"""Command-line interface: ``ingest``, ``query`` and ``stats``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from obsidian_rag import __version__
from obsidian_rag.config import RagConfig, load_config
from obsidian_rag.rag import RagError, RagIndex

EMBEDDERS = ["hashing", "tfidf", "sentence-transformers"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obsidian-rag",
        description="Offline retrieval-augmented search over a folder of Markdown notes.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=None, help="Optional TOML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Build an index from Markdown notes.")
    ingest.add_argument("notes", help="Folder containing Markdown notes.")
    ingest.add_argument("--index", default=None, help="Index directory (default: .obsidianrag).")
    ingest.add_argument("--embedder", default=None, choices=EMBEDDERS)
    ingest.add_argument("--max-chars", type=int, default=None, help="Max characters per chunk.")
    ingest.add_argument("--overlap", type=int, default=None, help="Character overlap on splits.")
    ingest.add_argument("--hashing-dim", type=int, default=None, help="Hashing vector size.")
    ingest.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    ingest.set_defaults(func=cmd_ingest)

    query = subparsers.add_parser("query", help="Search an existing index.")
    query.add_argument("question", help="Natural-language question.")
    query.add_argument("--index", default=None, help="Index directory (default: .obsidianrag).")
    query.add_argument("-k", "--top-k", type=int, default=None, help="Number of passages.")
    query.add_argument("--min-score", type=float, default=None, help="Minimum cosine score.")
    query.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    query.set_defaults(func=cmd_query)

    stats = subparsers.add_parser("stats", help="Show index statistics.")
    stats.add_argument("--index", default=None, help="Index directory (default: .obsidianrag).")
    stats.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    stats.set_defaults(func=cmd_stats)
    return parser


def _resolve_config(args: argparse.Namespace) -> RagConfig:
    return load_config(getattr(args, "config", None))


def cmd_ingest(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    index_dir = args.index or config.index_dir
    embedder = args.embedder or config.embedder
    max_chars = args.max_chars if args.max_chars is not None else config.max_chars
    overlap = args.overlap if args.overlap is not None else config.overlap

    embedder_kwargs: dict[str, Any] = {}
    if embedder == "hashing":
        embedder_kwargs["dim"] = args.hashing_dim or config.hashing_dim

    index = RagIndex.build(
        args.notes,
        embedder=embedder,
        max_chars=max_chars,
        overlap=overlap,
        **embedder_kwargs,
    )
    manifest = index.save(index_dir)
    result = {
        "index": str(Path(index_dir)),
        "embedder": embedder,
        "dim": index.store.dim,
        "chunks": index.store.size,
        "sources": len(manifest["sources"]),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Indexed {result['chunks']} chunks from {result['sources']} notes "
            f"into {result['index']} ({result['embedder']}, dim={result['dim']})."
        )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    index_dir = args.index or config.index_dir
    top_k = args.top_k if args.top_k is not None else config.top_k
    min_score = args.min_score if args.min_score is not None else config.min_score

    index = RagIndex.load(index_dir)
    result = index.answer(args.question, top_k=top_k, min_score=min_score)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if not result["citations"]:
        print("No relevant passages found.")
        return 0
    print(result["answer"])
    print("\nCitations:")
    for citation in result["citations"]:
        location = citation["source"]
        if citation["heading"]:
            location += f" > {citation['heading']}"
        print(
            f"  [{citation['index']}] {location} "
            f"(lines {citation['start_line']}-{citation['end_line']}, "
            f"score {citation['score']:.3f})"
        )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    index_dir = args.index or config.index_dir
    stats = RagIndex.load(index_dir).stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0
    print(f"Index:    {Path(index_dir)}")
    print(f"Embedder: {stats['embedder']} (dim={stats['dim']})")
    print(f"Chunks:   {stats['chunks']}")
    print(f"Sources:  {len(stats['sources'])}")
    for source in stats["sources"]:
        print(f"  - {source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RagError, FileNotFoundError, NotADirectoryError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
