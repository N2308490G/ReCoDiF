# Inputs for the manuscript analysis

## Download the original dataset

- [Dataset page (Zenodo, version 1)](https://zenodo.org/records/14947500)
- [Download the complete ZIP (286.5 MB)](https://zenodo.org/api/records/14947500/files/Mashhadi%20et%20al%20(T1-T2%20Data).zip/content)
- [Permanent DOI](https://doi.org/10.5281/zenodo.14947500)

The archive contains borehole and laboratory NMR T1–T2 raw measurements and
Excel files with geological tags. Save/extract it under `data/raw/` (excluded
from Git). For the B10 examples, the archive paths are
`Mashhadi et al (T1-T2 Data)/BNMR_Data/B10_6-4/` (7.0 m) and
`Mashhadi et al (T1-T2 Data)/BNMR_Data/B10_7-7/` (8.5 m).
Prepare model-ready NPZ cases and fold manifests under `inputs/` using the
[input schema](../docs/MODEL_IMPLEMENTATION.md#input-format).

The directories below already contain the CSV inputs used by the plotting
commands. Raw archive files must be converted to the documented input schema
before model training; the archive itself is not a ready-made training NPZ.

| Directory | Use |
|---|---|
| `observed/` | B10 acquisition maps at 7.0 and 8.5 m, record metadata and the existing source-file mapping |
| `field/` | The score, spectrum, predictive-interval, ablation, sensitivity, runtime and training-curve CSVs consumed by the B10 plotting/table programs |
| `controlled/` | Known-spectrum reconstruction, regional-law, conditional-allocation and two-amplitude fusion inputs |
| `settings/` | Four supplied configuration tables used for settings and controlled-score tabulation |

Only inputs consumed by the released programs and the B10 source metadata are
included. Numerical cells, column names and identifiers are preserved from the
supplied materials. Larger unused draw archives are not needed for these
figure/table commands and are not part of this package.

The B10 observations derive from the public archive:

S. R. Mashhadi, K. Keating, M. Müller-Petke, S. Costabel, T. Hiller,
A. Ospina Parra, and D. Grombacher. *T1–T2 NMR data of unconsolidated geological
materials*, version 1 (2025), [doi:10.5281/zenodo.14947500](https://doi.org/10.5281/zenodo.14947500).
The source archive is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The observation CSVs contain the selected original echo positions used in the
manuscript acquisition panels. The original full archive is available from
the DOI above.

`B10_6-4` denotes 7.0 m and `B10_7-7` denotes 8.5 m. Recovery/echo units are
explicit in CSV column names. Spectral CSV coordinates use `log10(T/s)`;
plotting converts axis labels to milliseconds without changing cell masses.

The `field/` and `controlled/` result groups are distinct experimental settings.
The readers keep their naming conventions and apply only the manuscript's
display-label mappings. No model fitting or benchmark execution is performed
by the plotting commands.
