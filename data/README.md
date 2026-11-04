# External data preparation

Observations, precomputed feature grids, maps and weights are excluded from this source release. No automatic downloader or data redistribution permission is provided.

```text
data/
  sio/{faa,vgg,east,north}/z_<raw_region>.csv
  swot/{faa,vgg,east,north}/z_<raw_region>.csv
  ship/single/<raw_region>.csv
  <region>/standard_<component>_<part>.csv
  <region>/swot_<component>_<part>.csv
  dataset_splits/split_seed_42/<region>/{train,val,test}.csv
  dataset_splits/split_seed_42/<region>/dataset_summary.json
  result/
```

Components: `faa`, `vgg`, `east`, `north`. Parts: `common`, `distinct`. Each common-cohort experiment requires all 16 precomputed CDPA grids. `standard` is the SIO-side decomposition. The full decomposition generator is not included. SIO FAA accepts the historical `grav` folder as a fallback.

| Study region | Raw tile |
|---|---|
| -165_-160_30_35 | -170_-160_30_40 |
| 131_136_1_6 | 130_140_0_10 |
| 170_175_31_37 | 170_180_30_40 |

Grid CSVs use latitude as the first column and longitude in the header. Coordinates must be finite and strictly monotonic. Cropped feature grids must have identical coordinates. Ship CSVs require `longitude`, `latitude`, and `bathymetry`; `survey_name` supports profile plots. Bathymetry is elevation in metres, negative underwater. Traditional baselines assume FAA in mGal; verify original product metadata.

Split CSVs are observation manifests, not CNN patch tensors. They preserve source row IDs, grid indices, membership and order. Export them from existing experimental split files using `export_dataset_splits.py`; inspect `--help`. The split manifest includes `sample_index`, `ship_row_id` (zero-based original CSV data row), `split`, `split_order`, `region`, `grid_row`, `grid_col`, `grid_latitude`, and `grid_longitude`. Preserve record order and do not merge or deduplicate held-out observations.

Before publication, supply product identifiers, versions, access links, units, preprocessing and redistribution terms. Filenames do not establish provenance. If data cannot be shared, provide documented access and preprocessing instructions with hashes.
