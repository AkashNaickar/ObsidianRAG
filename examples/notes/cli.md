# Command-line interface

The `obsidian-rag` command has three subcommands. Each one can print JSON with
`--json` for scripting.

## ingest

`obsidian-rag ingest ./notes` walks the folder, chunks every Markdown file, and
writes an index directory. `--embedder` selects the backend.

## query

`obsidian-rag query "how does chunking work"` loads the index and prints the
top passages plus a citation list with file, heading, and line numbers.

## stats

`obsidian-rag stats` reports the embedder, vector dimension, chunk count, and
the indexed source files.
