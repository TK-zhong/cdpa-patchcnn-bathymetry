# Features and methods

Four input components are FAA (`faa`), vertical gravity gradient (`vgg`), and the east/north components (`east`, `north`). Confirm the physical definitions and units with the input products; filenames alone are not metadata.

| Experiment | Inputs | Encoder |
|---|---|---|
| raw8 / arch_raw8 | Four SIO and four SWOT grids | Joint multi-scale CNN |
| cdpa12 / arch_full | Four averaged common grids, four SIO distinct grids, four SWOT distinct grids | Joint multi-scale CNN |
| cdpa16 | Both sources' common and distinct grids, 16 total | Joint multi-scale CNN |
| cdpa16_split | The same 16 CDPA grids | Separate common/distinct encoders |
| raw8_matched | Original eight grids | Width adjusted to approximate the split encoder's parameter count |
| arch_raw12 | Original eight grids plus four source-wise means | Joint multi-scale CNN |

The main model uses independent 9x9, 17x17 and 25x25 branches, input SE attention, convolutional residual blocks, and concatenated features for regression. `arch_same17` uses three independent 17x17 branches, not a single branch. `arch_no_skip` removes residual additions while retaining convolutional depth; `arch_no_se` removes SE attention. Read `gravity_models.py` for the executable definitions.

Precomputed CDPA features are required. The training repository does not implement the complete decomposition algorithm. Common plus distinct reconstruction diagnostics are available in `audit_cdpa_features.py`.

## Traditional regional baselines

FAA is assumed to be in mGal; bathymetry is elevation in metres, positive upward. GGM uses the infinite-slab relation with a training-control-point background gravity grid and local inverse-distance interpolation. SAS-style inversion uses a band-limited downward continuation, a regional scale fitted on training observations, and a training-derived long-wave bathymetry background. Parker uses a sixth-order, damped, filtered Parker–Oldenburg iteration and restores a training-derived long-wave background.

These are regional variants, not full reproductions of the original authors' software. The nonlinear coupling between the restored background and Parker relief is not fully verified. Converged updates alone do not prove physical consistency. The existing tests check slab units/sign, the linear spectral response and GGM control recovery; a nonlinear forward/inverse closure test remains necessary for stronger physical claims.

Density/filter candidates are selected on validation RMSE. No GEBCO grid is used to build the background. Duplicate training cells are aggregated by median; evaluation retains all observations. A pure sounding-only interpolation baseline and shared-cell versus unseen-cell evaluation are recommended before interpreting gravity-specific gains.

References: [GGM study](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.636246/full), [Smith and Sandwell description](https://topex.ucsd.edu/marine_topo/text/topo.html), [Parker–Oldenburg bathymetry inversion](https://academic.oup.com/gji/article/216/2/840/5159468).


## Numerical settings

GGM uses `a=2*pi*G*density*1e5` mGal/m, `D=min(training cell medians)`, training backgrounds `g_reg=g-a*(z-D)`, and prediction `z=D+(g-IDW(g_reg))/a`. IDW uses 12 nearest training cells and inverse squared planar distance, with exact recovery at controls. Density candidates are 1000/1400/1670/2000/2400 kg/m3.

For spectral methods, `k=2*pi*hypot(fx,fy)`. The band filter is `exp(-(k*s/(2*pi))**4)*(1-exp(-(k*L/(2*pi))**4))`. The reference lowpass is `exp(-0.5*(k*L/(2*pi))**4)`. Reflective padding is 32 cells. Grid distances use a regional-mean-latitude equirectangular approximation. Downward continuation exponents are capped at 30.

SAS fits a positive scale plus an offset on original training observations. Short wavelengths are 10/15/25 km and long wavelengths 80/160/240 km. Parker uses density candidates 1400/1670/2000 kg/m3, short wavelengths 15/25/40 km and long wavelengths 80/160 km. It starts with zero relief, retains six terms, uses damping 0.25 and at most 100 iterations. Update RMS must fall below 0.05 m. Unconverged/non-finite candidates and absolute relief exceeding 20000 m are rejected.

## Controlled comparisons

`arch_full`, `arch_same17`, `arch_no_skip` and `arch_raw12` have 5,227,910 parameters; `arch_no_se` has 5,227,649, and `arch_raw8` 5,220,863. All architecture experiments use a common valid cohort and require full 25x25 context. MACs count convolution/linear multiply-accumulate operations only, not normalization, activation, pooling or elementwise gates. Equal parameters do not establish equal computation.
