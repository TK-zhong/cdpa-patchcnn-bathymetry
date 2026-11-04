"""Regional, train-constrained GGM, Smith-Sandwell-style and Parker-Oldenburg baselines.

Elevation positive upward (ocean negative); FAA in mGal; SI internally.
See TRADITIONAL_METHODS.md for the precise regional variants and limitations.
"""
import math
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.spatial import cKDTree

G = 6.67430e-11


class IDW:
    """Reusable local interpolation weights; inputs are training grid cells only."""
    def __init__(self, cells, shape, spacing, neighbors=12):
        self.cells, self.shape = np.asarray(cells), shape
        query = np.indices(shape).reshape(2, -1).T * spacing
        distance, self.index = cKDTree(self.cells * spacing).query(query, k=min(neighbors, len(cells)))
        if distance.ndim == 1:
            distance, self.index = distance[:, None], self.index[:, None]
        weights = 1 / np.maximum(distance, 1e-6)**2
        exact = distance[:, 0] < 1e-6
        weights[exact] = 0
        weights[exact, 0] = 1
        self.weights = weights / weights.sum(axis=1, keepdims=True)

    def __call__(self, values):
        return (np.asarray(values)[self.index] * self.weights).sum(axis=1).reshape(self.shape)


class Spectrum:
    def __init__(self, shape, spacing, pad=32):
        self.shape, self.pad = shape, pad
        ny, nx = np.array(shape) + 2*pad
        self.k = 2*np.pi*np.hypot(fftfreq(ny, spacing[0])[:, None], fftfreq(nx, spacing[1])[None, :])

    def transform(self, data):
        return fft2(np.pad(data, self.pad, mode='reflect'))

    def crop(self, data):
        p = self.pad
        return data[p:p+self.shape[0], p:p+self.shape[1]]

    def lowpass(self, data, wavelength):
        return self.crop(ifft2(self.transform(data)*np.exp(-0.5*(self.k*wavelength/(2*np.pi))**4)).real)

    def band(self, short, long):
        return np.exp(-(self.k*short/(2*np.pi))**4) * (1-np.exp(-(self.k*long/(2*np.pi))**4))


def metrics(pred, true):
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError('Non-finite evaluation values')
    error = pred-true
    mse = float(np.mean(error**2))
    total = np.sum((true-true.mean())**2)
    return dict(N=len(true), MSE=mse, RMSE=math.sqrt(mse), MAE=float(np.mean(abs(error))),
                R2=float(1-np.sum(error**2)/total) if total else None)


def ggm(gravity, z_train, cells, interpolator, density):
    coefficient = 2*np.pi*G*density*1e5  # mGal / metre
    datum = float(z_train.min())
    g_train = gravity[tuple(cells.T)]
    regional = interpolator(g_train-coefficient*(z_train-datum))
    return datum+(gravity-regional)/coefficient


def sas(gravity, reference, spectrum, train_rows, train_cols, train_z, short, long):
    """Linear downward continuation, train-calibrated admittance, restored long waves."""
    background = spectrum.lowpass(reference, long)
    z0 = max(100., -float(train_z.mean()))
    band = spectrum.band(short, long)
    signal = spectrum.crop(ifft2(spectrum.transform(gravity-gravity.mean()) *
                                band*np.exp(np.minimum(spectrum.k*z0, 30))).real)
    x = signal[train_rows, train_cols]
    y = train_z-background[train_rows, train_cols]
    # A single regional scale and offset, fitted on original train observations.
    design = np.column_stack((x, np.ones(len(x))))
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    if scale <= 0:
        raise ValueError('Non-positive gravity/topography admittance')
    return background+scale*signal+offset, dict(scale_m_per_mgal=float(scale), offset_m=float(offset), z0_m=z0)


def parker_forward_spectrum(h, k, z0, density, terms=6):
    """Parker expansion for positive-upward relief above a negative reference level."""
    series = np.zeros(h.shape, dtype=complex)
    for n in range(1, terms+1):
        series += k**(n-1)*fft2(h**n)/math.factorial(n)
    return 2*np.pi*G*density*1e5*np.exp(-k*z0)*series


def parker(gravity, reference, spectrum, mean_z, density, short, long, max_iter=100):
    """Filtered, damped Oldenburg iteration of the sixth-order Parker series."""
    z0 = max(100., -mean_z)
    background = spectrum.lowpass(reference, long)
    band = spectrum.band(short, long)
    target = spectrum.transform(gravity-gravity.mean())
    linear = target * np.exp(np.minimum(spectrum.k*z0, 30))/(2*np.pi*G*density*1e5)
    h = np.zeros(spectrum.k.shape)
    change = float('inf')
    for iteration in range(1, max_iter+1):
        nonlinear = np.zeros(h.shape, dtype=complex)
        for n in range(2, 7):
            nonlinear += spectrum.k**(n-1)*fft2(h**n)/math.factorial(n)
        proposal = ifft2(band*(linear-nonlinear)).real
        updated = .75*h+.25*proposal
        change = float(np.sqrt(np.mean((updated-h)**2)))
        if not np.isfinite(updated).all() or np.max(abs(updated)) > 20000:
            raise ValueError('Parker iteration diverged')
        h = updated
        if change < .05:
            break
    if change >= .05:
        raise ValueError('Parker iteration did not converge within tolerance')
    return background+spectrum.crop(h), dict(iterations=iteration, last_update_rms_m=change, z0_m=z0,
                                             terms=6, damping=.25, tolerance_m=.05)
