# fragExtracor_pipeline

Offline pipeline for:

1. LLM Step-1 keyword extraction from attack `instruction`
2. Rule-based generation of `masked_query` and `carrier_query`

## Input Format

Typical attack rows include:

| Field | Purpose |
|------|------|
| `id` | Case identifier |
| `instruction` | Attack-side instruction text (LLM keyword scan target) |
| `host_instruction` | Host/WebShop benign instruction used as `carrier_query` prefix |
| `profile` | User profile (copied through by extractor outputs) |
| `host_fix_number` / `Instruction_fix_number` | Dataset-level identifiers used by upper-layer runners |

Optional `sensitive_fragments` can exist as metadata, but pipeline extraction uses LLM output.

## Script

### `frag_mask_pipeline.py` (one-pass)

Runs LLM extraction and masking/carrier construction in one pass.
All non-derived fields are copied to output; `fragment`, `masked_query`, and `carrier_query` are regenerated.

```bash
cd /Users/zixinrao/Desktop/rap-fragment/Attacks-on-Agents/webshop/fragExtracor_pipeline

python3 frag_mask_pipeline.py \
  --input dataset_input.json \
  --output output.json \
  --model gpt-4o

# Optional: --limit N  --verbose  --no-progress
```

This pipeline now replaces the previous split workflow and always performs:
- Step-1 LLM extraction (`fragment`)
- `masked_query` construction
- `carrier_query` construction

## Output Semantics

- `masked_query`: replaces extracted fragments in `instruction` with `<>` (longest-first, one replacement each)
- `carrier_query`: `host_instruction + " ." + <chunk><chunk>...` based on built-in fragment chunking rules

## Notes

- `Instruction_fix_number` is consumed by `webshop/main.py` for `fixed_<n>` WebShop session routing.
- If many rows have empty `fragment`, inspect the Step-1 prompt and parser in `frag_mask_pipeline.py`.
- Rows without valid `instruction` are rejected with a validation error.
