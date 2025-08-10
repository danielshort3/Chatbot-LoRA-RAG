# Changelog

## [Unreleased]

### Added
- `--compare` CLI flag to launch Compare Mode. This flag is backward compatible and does not change behaviour when omitted.
- `merge_lora.py` script to merge the adapter into a 4‑bit quantised model.

### Changed
- Cap inference context retrieval at three snippets to match training limits.

