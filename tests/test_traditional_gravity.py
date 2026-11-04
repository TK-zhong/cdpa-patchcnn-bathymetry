import unittest
import numpy as np
from scipy.fft import fft2, ifft2
from traditional_gravity import G, IDW, Spectrum, ggm, parker_forward_spectrum


class PhysicalChecks(unittest.TestCase):
    def test_slab_units_and_sign(self):
        # An infinite slab of 100 m relief at 1670 kg/m3 gives ~7 mGal.
        h = np.full((16, 16), 100.)
        k = np.zeros(h.shape)
        result = ifft2(parker_forward_spectrum(h, k, 4000., 1670.)).real
        np.testing.assert_allclose(result, 2*np.pi*G*1670*100*1e5)

    def test_linear_sinusoid_transfer(self):
        x = np.arange(64)*2000.
        h = np.tile(10*np.cos(2*np.pi*x/128000), (64, 1))
        spec = Spectrum(h.shape, (2000., 2000.), pad=0)
        response = parker_forward_spectrum(h, spec.k, 4000., 1670., terms=1)
        expected = fft2(h)*2*np.pi*G*1670*1e5*np.exp(-spec.k*4000.)
        np.testing.assert_allclose(response, expected, atol=1e-10)

    def test_ggm_recovers_control_cells(self):
        cells = np.array([[2, 2], [4, 4], [6, 6]])
        z = np.array([-4000., -2500., -1000.])
        gravity = np.arange(81.).reshape(9, 9)
        interp = IDW(cells, gravity.shape, (2000., 2000.))
        predicted = ggm(gravity, z, cells, interp, 1670.)
        np.testing.assert_allclose(predicted[tuple(cells.T)], z)


if __name__ == '__main__':
    unittest.main()
