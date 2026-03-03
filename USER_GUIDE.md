# Embedding Based Internal Linking Tool - User Guide

## Overview
This tool has two workflows:

1. Inlinks analysis
- Input: Screaming Frog All Inlinks export (`.csv`, `.csv.gz`, `.zip`)
- Purpose: remove non-contextual links and summarise contextual internal linking

2. Embedding recommendations
- Input: embeddings export + contextual baseline links
- Purpose: generate new internal link opportunities while excluding links that already exist

## How to export inputs from Screaming Frog

## A) All Inlinks CSV (for Inlinks analysis)
1. Crawl your site in Screaming Frog.
2. Go to `Bulk Export -> Links -> All Inlinks`.
3. Export as `.csv`.
4. Optional: compress to `.csv.gz` or `.zip` before upload.

## B) Embeddings CSV (for Embedding recommendations)
1. Run a crawl with AI embedding extraction enabled for page content.
2. Confirm embedding values are present in `Extract embeddings from page content`.
3. Export a CSV that includes:
- URL column (for example `Address`)
- embedding vector column (for example `Extract embeddings from page content`)

## Workflow 1: Inlinks analysis

1. Open the app and choose `Inlinks analysis`.
2. Upload your All Inlinks export.
3. Keep `Contextual mode` enabled (recommended default).
4. Optional: adjust filters (parameter, pagination, anchors, external source domain).
5. Click `Run analysis`.
6. Review:
- `Top destinations`
- `Filter impact`
- `Performance Metrics`
7. Download:
- `top destinations CSV`
- `full destination summary CSV`
- `full filtered rows CSV` (if enabled)

## Workflow 2: Embedding recommendations

1. Switch to `Embedding recommendations`.
2. Upload embeddings CSV from Screaming Frog (`Extract embeddings from page content`).
3. Set contextual baseline:
- Use last filtered export from Workflow 1, or
- Upload filtered inlinks CSV (`Source` + `Destination` required)
4. Optional: upload `Focus URLs` CSV for destination targeting and prioritisation metric.
5. Set controls:
- `Recommendations per page`
- `Minimum similarity`
- `Max pages to process`
6. Click `Generate embedding recommendations`.
7. Review and download:
- `Link Opportunities` (combined view)
- `New Recommendations`
- `Skipped Existing Links`

## Outputs explained

## Top destinations
- Destination pages receiving the most contextual links after filters
- Includes top/secondary anchor usage percentages

## Filter impact
- Links removed by each filter
- Helps users understand what rules changed the dataset most

## Link Opportunities
- Unified output for actioning and audit
- Key columns:
- `Source_URL`
- `Recommended_Target_URL`
- `Similarity`
- `Already_Exists`
- `Reason`
- `Opportunity_Score`
- Optional `Priority_Metric (...)`

## Opportunity Score
- If numeric priority metric exists:
- `0.7 * Similarity + 0.3 * normalized_metric`
- Else:
- `Similarity`
- If `Already_Exists=True`: score forced to `0`

## Recommended operating pattern

1. Run Inlinks analysis and export filtered contextual links.
2. Use that filtered export as contextual baseline in Embedding recommendations.
3. Optionally upload focus destination URLs with a business metric.
4. Prioritise rows where:
- `Already_Exists=False`
- highest `Opportunity_Score`

## Troubleshooting

- ZIP contains `__MACOSX` metadata: supported; app ignores those files.
- Missing `Source`/`Destination`: re-export All Inlinks with required columns.
- No embedding recommendations:
- lower minimum similarity
- increase max pages
- check overlap between embeddings URLs and baseline/focus URLs
- Focus URLs not matching:
- ensure URL format/domain matches embeddings dataset after normalization.
