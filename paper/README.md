# Compiling the paper

**Status: this LaTeX has not been compiled yet.** The machine it was written on
has no LaTeX, and a portable compiler could not be downloaded without the
owner's permission. A structural check ran instead (`check_tex.py`), which is
not a substitute for compiling.

With TeX Live or MiKTeX:

```
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

With Tectonic: `tectonic main.tex`. With Overleaf: upload this folder and set
`main.tex` as the main document.

## Regenerating tables and figures

Tables in `tables/` and figures in `figures/` are generated, so do not edit them
by hand. From the repository root:

```
python -m fte_fl.eval.paper_artifacts \
    --femnist results/runs/m6/femnist_main_lr0.1 \
    --control results/runs/m6_windows/femnist_stratified_control               results/runs/m6_windows/femnist_stratified_control_rerun \
    --speech bcres=results/runs/m6_windows/speech_main_bcres \
    --bench results/runs/m6_windows/protocol_bench \
    --out paper
```

`tables/missing.txt` lists any group that failed a check, and `tables/values.json`
holds the values used.

Before submission, complete the author block in `main.tex`.
