# Figure provenance

`cambria-2023-figure-1.png` reproduces **Figure 1, p. 4** from:

Cambria, E., Malandri, L., Mercorio, F., Mezzanzanica, M., & Nobani, N. (2023).
*A survey on XAI and natural language explanations*. Information Processing &
Management, 60(1), 103111. [DOI](https://doi.org/10.1016/j.ipm.2022.103111).

The figure and its original caption were extracted from the published PDF supplied
for this repository update. Only surrounding page material was cropped away;
labels, arrows, colours and caption were not altered. The article's notice is
**© 2022 Elsevier Ltd. All rights reserved.** The figure is third-party material,
excluded from the repository's MIT and CC BY 4.0 grants; this record does not
assert a separate general reuse license. The full PDF is not redistributed here.

- Input PDF SHA-256: `717a2dbdffcc410bcbf6f63819fa16d16b4e3ebfde0e202aed8dc2023eb39392`
- Output PNG SHA-256: `7087134236c6d46aa7f198b776d483afda640f4e1c5a92c38a8580b4cfbce65b`
- Output dimensions: 1470 × 490 pixels.
- Source pages: PDF page 4, printed page 4.

Extraction with Poppler (the input PDF must be supplied separately):

```bash
pdftoppm -f 4 -singlefile -r 240 -x 185 -y 150 -W 1470 -H 490 -png \
  explainable-artificial-intelligence.pdf cambria-2023-figure-1
```

The [notebook](../Dimensions.ipynb) interprets the figure and cites the relevant
paper sections. Its explanation tables and worked examples are separate teaching
material, not altered versions of the figure or reproductions of survey results.
