English | [Русский](README.ru.md)

# Python-SDM
Species Distribution Modelling (SDM) library for Python.

Python-SDM finds patterns between where a species has been observed and the environmental conditions at those locations, then projects a continuous probability-of-occurrence surface over a region of interest (current, future and past time periods, optionally at monthly resolution).

A hosted version runs online, with a large, pre-aligned set of environmental predictors already in place, so a model can be fitted for any of several thousand already-supported species without any local setup:
- Demo: https://wingeds.world/en/sdm (English) / https://wingeds.world/sdm (Russian)
- Documentation: https://wingeds.world/en/sdmdoc (English) / https://wingeds.world/sdmdoc (Russian)

Birds, mammals, reptiles, insects and plants are all supported online.

A general introduction to the SDM method behind this library is available (in Russian) at https://habr.com/ru/articles/983756/.

## Methodological notes

For readers familiar with the SDM/ENM literature, a few implementation choices are worth calling out explicitly:

- **Accessibility (the "M" in the BAM diagram, Soberón & Peterson 2005; Barve et al. 2011).** Rather than a flat Euclidean buffer around known occurrences, the accessible area can be computed as a cost-distance (friction surface) via the Fast Marching Method, with terrain slope, elevation limits and water bodies acting as barriers of configurable strength. Several decay functions from the occurrence buffer to zero are available (buffer, linear, sigmoid, exponential, inverse-quadratic, Gaussian).
- **Sample selection bias correction.** Background/pseudo-absence points can be drawn with a probability proportional to the pooled, smoothed density of all GBIF occurrence records of the same taxonomic class in the region, following the target-group background idea of Phillips et al. (2009), rather than purely at random.
- **Evaluation.** Besides ROC AUC, TSS and Cohen's kappa, the continuous Boyce index (Hirzel et al., 2006) is reported, which does not require true absence data.
- **Per-predictor niche diagnostics.** For every environmental layer, the distribution of values at occurrence points is compared to the distribution over the full background using Schoener's D (Schoener, 1968; cf. Warren et al., 2008), alongside summary statistics (median, percentiles, a standardized difference of medians).
- **Models.** Random Forest and XGBoost (via scikit-learn / xgboost); a standard GLM (logistic regression with linear and quadratic terms, L2 regularization strength chosen by cross-validation on ROC AUC — quadratic terms let it represent a unimodal, optimum-shaped response rather than only a monotonic one, cf. ter Braak & Looman, 1986); and MaxEnt, adapted from the modelling core of [elapid](https://github.com/earth-chris/elapid) (linear/hinge/threshold/product features, L1-regularized logistic regression, raw/logistic/cloglog output calibration) with its raster/vector I/O stripped out, since this library already handles that itself. Only Random Forest and XGBoost currently support future/past projections; GLM and MaxEnt are fit and evaluated for the present only.

## Data sources

- Observation data: GBIF.org, iNaturalist.org, eBird.org, wingeds.world, plantarium.su, and others.
- Environmental predictors: WorldClim (temperature, precipitation), SRTM (terrain, elevation), OpenStreetMap (rivers and roads), among others.

## Local installation and a minimal run

**To run Python-SDM locally:**

1. Download the library into a folder, e.g. `python_sdm`.

2. Install Python:
```
yum install python3 python3-pip libxcb mesa-libGL # CentOS, RHEL
# or
apt install python3 python3-pip libxcb1 libgl1 # Ubuntu, Debian
```
3. Install the Python dependencies:
```
pip install pandas numpy scipy matplotlib scikit-learn xgboost opencv-python rasterio contextily pyproj Pillow affine joblib
```

**The library is technically ready to run. Next, gather the input data: predictor layers and observation records.**

4. Create a folder `python_sdm\input_predictors` for the predictor layers. It can hold up to five subfolders:
- `python_sdm\input_predictors\static` — time-invariant predictors (e.g. elevation, WorldClim height), usable for every time period;
- `python_sdm\input_predictors\dynamic_current` — predictors for the present (e.g. the WorldClim BIOxx variables). If you model the future, files with matching names must also exist in the next folder;
- `python_sdm\input_predictors\dynamic_predictable` — predictors for future projections (e.g. WorldClim BIOxx under SSP370). Layout: one subfolder per period (e.g. `2041-2060`), each containing one subfolder per scenario (e.g. `ssp370`) — i.e. `dynamic_predictable\2041-2060\ssp370\`;
- `python_sdm\input_predictors\dynamic_past` — predictors for past reconstructions (e.g. CHELSA TraCE21k). Layout is flatter: one subfolder per period, no scenarios (e.g. `dynamic_past\02-180\`), plus one mandatory `dynamic_past\learning\` subfolder holding present-day predictors in the same format as the other past periods — a dedicated model is trained on it for past projections;
- `python_sdm\input_predictors\dynamic_monthly` — only if monthly projections are enabled (`MODEL_SEASON: 1`): one subfolder per month (`1` … `12`).

Predictors must be GeoTIFF files. If a file has several bands, only the first is used.

For a minimal run, download `WorldClim elevation` (https://www.worldclim.org/data/worldclim21.html, `30 seconds` resolution) into `python_sdm\input_predictors\static`, and the `Bioclimatic variables` into `python_sdm\input_predictors\dynamic_current`.

5. Copy the contents of `python_sdm\examples` (the `data` folder and `example_minimal.py`) into the root of `python_sdm` (next to the `sdm` and `input_predictors` folders).

6. From that folder, run:
```
python example_minimal.py
```
This fits an SDM for the peregrine falcon (*Falco peregrinus*) over the Altai region.

7. Observation data for any other species can be placed into `python_sdm\data`; it can be downloaded from GBIF.org. A few minimal examples already ship in `examples\data`.

## License

MIT — see [LICENSE](LICENSE). The MaxEnt implementation in `sdm/utils/elapid_core/` is adapted from [elapid](https://github.com/earth-chris/elapid) (MIT, © 2021 Christopher Anderson); see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full notice.
