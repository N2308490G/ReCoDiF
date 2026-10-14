# Inputs for the manuscript analysis

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
A. O. Ospina, and M. Grånbæk. *T1–T2 NMR data of unconsolidated geological
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
