import unittest
import numpy as np
import torch
from torch import nn

from CDPA_split import build_experiment_features, COMPONENTS
from gravity_models import (make_model, architecture_config, parameter_count,
                            conv_linear_macs, ARCHITECTURE_EXPERIMENTS, ResBlock)


class ArchitectureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_controls(self):
        counts = {name: parameter_count(make_model(name, 12)) for name in
                  ('arch_full', 'arch_same17', 'arch_no_skip', 'arch_raw12')}
        self.assertEqual(len(set(counts.values())), 1)
        self.assertEqual(counts['arch_full'], 5227910)

    def test_raw12_preserves_raw_features_and_only_adds_means(self):
        raw = {f'{source}_{c}': np.full((3, 4), index+1, dtype=np.float32)
               for index, (source, c) in enumerate((s, c) for s in ('sio', 'swot') for c in COMPONENTS)}
        features, names = build_experiment_features({}, raw, 'arch_raw12')
        self.assertEqual(features.shape, (12, 3, 4))
        for i, key in enumerate([f'{s}_{c}' for s in ('sio', 'swot') for c in COMPONENTS]):
            np.testing.assert_array_equal(features[i], raw[key])
        for i, c in enumerate(COMPONENTS):
            np.testing.assert_array_equal(features[8+i], (raw[f'sio_{c}']+raw[f'swot_{c}'])/2)
        self.assertEqual(len(names), 12)

    def test_no_skip_preserves_weights_and_depth(self):
        full, plain = make_model('arch_full', 12), make_model('arch_no_skip', 12)
        self.assertEqual(list(full.state_dict()), list(plain.state_dict()))
        self.assertTrue(all(not m.use_skip for m in plain.modules() if isinstance(m, ResBlock)))
        block = ResBlock(2, use_skip=False).eval()
        x = torch.randn(2, 2, 5, 5)
        torch.testing.assert_close(block(x), torch.relu(block.conv(x)))

    def test_no_se_and_every_forward_backward(self):
        for name in (*ARCHITECTURE_EXPERIMENTS, 'arch_single17'):
            channels = 8 if name == 'arch_raw8' else 12
            model = make_model(name, channels)
            halfs = architecture_config(name)['halfs']
            if name == 'arch_no_se':
                self.assertTrue(all(isinstance(m, nn.Identity) for m in model.se_blocks))
            prediction = model([torch.randn(2, channels, 2*h+1, 2*h+1) for h in halfs])
            self.assertEqual(prediction.shape, (2,))
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_same_scale_cost_is_measured_and_mode_preserved(self):
        counts = {}
        for name in ('arch_full', 'arch_same17'):
            model = make_model(name, 12).train()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            counts[name] = conv_linear_macs(model, 12, architecture_config(name)['halfs'])
            self.assertTrue(model.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])
        self.assertGreater(counts['arch_full'], counts['arch_same17'])


if __name__ == '__main__':
    unittest.main()
