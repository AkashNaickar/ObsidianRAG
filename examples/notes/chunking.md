# Chunking

ObsidianRAG splits every note into chunks before embedding. A chunk is a
passage of text plus the source path, the heading trail, and the start and end
line numbers in the original file.

## Paragraph packing

Paragraphs are packed together up to `max_chars` characters. Packing never
crosses a heading boundary, so a chunk always belongs to a single heading.

## Long blocks and overlap

A paragraph longer than `max_chars` is split into overlapping windows. The
`overlap` setting keeps neighbouring windows sharing a suffix and prefix so a
match that straddles a split is still retrievable.

## Why headings matter

Because the heading trail is embedded alongside the passage, a query that
mentions a topic name can still match notes whose body text uses different
wording.
