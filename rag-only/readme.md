# Fetal cardiac screening RAG data

This folder is the **minimal export**. It contains only the active clinical RAG records, retrieval indexes, and 379 fetal cardiac USG scan images. It has no dashboard, full-text papers, per-paper review folders, or audit reports.

## Contents

- `records/`: JSONL views, findings, diseases, workflows, cases, media, evidence, relationships, and a compact `sources.jsonl` bibliography. Each clinical claim links to evidence IDs and a source location.
- `indexes/`: SQLite FTS5, deterministic descriptor vectors, and precomputed multilingual MiniLM text and CLIP image vectors. Model names, revisions, dimensions, row IDs and source-record hashes are in the index JSON files.
- `images/`: the 379 USG stills referenced by `records/media.jsonl`. The `asset_path` field is relative to this folder and the `sha256` field verifies each image.

Counts: 1520 clinical knowledge and image records, 1445 evidence entries, 1500 relationships, and 68 cited papers. Tables, study graphs, diagrams, and extracardiac-only images are absent from the active image set.

## Use

Open `indexes/kb.sqlite` with any SQLite client that supports FTS5. The `records_fts` table searches record labels and clinical text; its `id` joins to `records.id` and the JSONL entries. The `image_ids.json` and `text_ids.json` arrays define the row order of the vector files. The `.npy` files are deterministic descriptors. The `.f32` files are little-endian float32 arrays shaped as specified in their accompanying JSON files. To embed a new query with the semantic models, use the exact model and revision named there; the model weights are not included in this data-only archive.

All 379 images are locally usable for research inspection, but only 23 have rights marked `CLEAR`; 345 are restricted and 11 have uncertain rights. These flags are preserved in `records/media.jsonl`. Do not redistribute restricted or uncertain images without permission.

This is a source-linked research dataset, not a clinician-certified diagnostic reference. The full research archive retains four blocked paper reviews, so the corpus has not been declared complete. Clinical thresholds and imaging criteria require specialist review before clinical deployment.
