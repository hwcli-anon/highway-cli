# Anonymous public-release checklist

- [x] Project-owned code license selected and `LICENSE` added.
- [ ] IEEE IV anonymity policy rechecked from the current official call.
- [ ] Repository owner, profile, commit authors, issue templates, and links do
      not identify the authors or institution.
- [x] `python -m pip install -e .` succeeds in a new Python environment.
- [x] `python -m unittest discover -s tests -v` reports 88 passing tests.
- [x] Offline interaction-graph mock demo succeeds without network access.
- [ ] NADE doctor and one-episode mock closed-loop smoke test succeed when the
      separately licensed dependency is installed.
- [x] No API key, token, private URL, local absolute path, username, email,
      billing metadata, raw provider response, or private paper is present.
- [x] `python tools/check_public_tree.py` passes; identity terms and image
      metadata are also reviewed manually.
- [x] No milestone directory, large trace, cache, virtual environment, or model
      weight is present.
- [ ] README commands are re-run verbatim by a person who did not build the
      framework.
- [ ] Public files are compared with the manuscript's frozen protocol.
- [ ] An immutable release tag and file manifest are created after approval.
