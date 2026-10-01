# Security Policy

## Supported versions

ObsidianRAG is a small, pre-1.0 project. Security fixes are applied to the
latest release on the `main` branch.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Instead, use GitHub's private reporting flow:
[Report a vulnerability](https://github.com/AkashNaickar/ObsidianRAG/security/advisories/new).

Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal note folder or command is ideal),
- any suggested fix.

You can expect an acknowledgement within 7 days. Please allow time for a fix
and release before disclosing the issue publicly.

## Scope and design notes

- ObsidianRAG runs entirely on your machine. It does not send note contents to
  any remote service.
- The optional `sentence-transformers` backend downloads a model on first use;
  with the default `hashing` and `tfidf` embedders there is no network access.
- The tool reads Markdown files from a folder you point it at and writes an
  index directory. Treat note folders and index files as sensitive data and
  keep them out of version control (the default `.gitignore` excludes
  `.obsidianrag/`).
- Never commit secrets or private notes to this repository.
