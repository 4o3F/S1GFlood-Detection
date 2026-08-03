from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import DAMNet_New, WaterSegmentationHead


class TinyCTCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)

    def forward(self, pre, post):
        return self.projection(pre), self.projection(post)


class TinyTACE(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)

    def forward(self, tokens, temporal_tokens):
        batch, count, channels = tokens.shape
        side = int(count ** 0.5)
        feature_map = self.projection(tokens).transpose(1, 2).reshape(
            batch,
            channels,
            side,
            side,
        )
        return feature_map, temporal_tokens.mean(dim=1)


class TinyTDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, pre, post, semantic):
        logits = self.projection(torch.abs(pre - post))
        return F.interpolate(
            logits,
            scale_factor=8,
            mode='bilinear',
            align_corners=False,
        )


class TinyDAMNet(DAMNet_New):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = nn.Conv2d(3, 32, kernel_size=8, stride=8)
        self.token_len = 4
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, bias=False)
        self.tokenizer = True
        self.token_trans = False
        self.CTCA = TinyCTCA()
        self.decoder_projection = nn.Linear(32, 32)
        self.TACE_pre = TinyTACE()
        self.TACE_post = TinyTACE()
        self.TDF = TinyTDF()
        self.output_sigmoid = False
        self.water_head = WaterSegmentationHead()
        self.forward_single_calls = 0

    def forward_single(self, image):
        self.forward_single_calls += 1
        return self.backbone(image)

    def _forward_transformer_decoder(self, feature_map, tokens):
        decoded = feature_map.flatten(2).transpose(1, 2)
        decoded = self.decoder_projection(decoded)
        return feature_map, decoded


class AuxiliaryWaterHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TinyDAMNet().eval()
        self.image_a = torch.randn(2, 3, 256, 256)
        self.image_b = torch.randn(2, 3, 256, 256)

    def test_default_contract_remains_tensor(self):
        output = self.model(self.image_a, self.image_b)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 2, 256, 256))

    def test_production_constructor_registers_shared_water_head(self):
        arguments = SimpleNamespace(backbone='resnet')
        with patch('networks.ResNet', return_value=nn.Identity()):
            model = DAMNet_New(
                arguments,
                input_nc=3,
                output_nc=2,
                with_pos='learned',
            )
        self.assertIsInstance(model.water_head, WaterSegmentationHead)
        names = [name for name, _ in model.named_parameters()]
        self.assertTrue(any(name.startswith('water_head.') for name in names))

    def test_auxiliary_contract_has_named_full_resolution_logits(self):
        output = self.model(self.image_a, self.image_b, return_aux=True)
        self.assertEqual(
            set(output),
            {'change_logits', 'water_a_logits', 'water_b_logits'},
        )
        for logits in output.values():
            self.assertEqual(tuple(logits.shape), (2, 2, 256, 256))

    def test_auxiliary_branch_reuses_single_temporal_backbone_features(self):
        self.model.forward_single_calls = 0
        self.model(self.image_a, self.image_b, return_aux=True)
        self.assertEqual(self.model.forward_single_calls, 2)

    def test_auxiliary_request_does_not_change_main_logits(self):
        default = self.model(self.image_a, self.image_b)
        auxiliary = self.model(
            self.image_a,
            self.image_b,
            return_aux=True,
        )
        torch.testing.assert_close(default, auxiliary['change_logits'])

    def test_shared_head_is_temporally_swap_equivariant(self):
        forward = self.model(self.image_a, self.image_b, return_aux=True)
        swapped = self.model(self.image_b, self.image_a, return_aux=True)
        torch.testing.assert_close(
            forward['water_a_logits'],
            swapped['water_b_logits'],
        )
        torch.testing.assert_close(
            forward['water_b_logits'],
            swapped['water_a_logits'],
        )

    def test_exactly_one_water_head_is_registered(self):
        names = [name for name, _ in self.model.named_parameters()]
        self.assertTrue(any(name.startswith('water_head.') for name in names))
        self.assertFalse(any(name.startswith('water_head_a.') for name in names))
        self.assertFalse(any(name.startswith('water_head_b.') for name in names))

    def test_auxiliary_gradient_bypasses_temporal_path(self):
        output = self.model(self.image_a, self.image_b, return_aux=True)
        loss = (
            output['water_a_logits'].mean()
            + output['water_b_logits'].mean()
        )
        self.model.zero_grad()
        loss.backward()

        self.assertIsNotNone(self.model.backbone.weight.grad)
        self.assertIsNotNone(self.model.water_head.refine[0].weight.grad)
        bypassed_modules = (
            self.model.conv_a,
            self.model.CTCA,
            self.model.decoder_projection,
            self.model.TACE_pre,
            self.model.TACE_post,
            self.model.TDF,
        )
        for module in bypassed_modules:
            for parameter in module.parameters():
                self.assertIsNone(parameter.grad)

    def test_legacy_instance_keeps_default_change_contract(self):
        del self.model.water_head
        output = self.model(self.image_a, self.image_b)
        self.assertIsInstance(output, torch.Tensor)
        with self.assertRaisesRegex(RuntimeError, 'does not contain'):
            self.model(self.image_a, self.image_b, return_aux=True)

    def test_headless_legacy_checkpoint_round_trip(self):
        del self.model.water_head
        expected = self.model(self.image_a, self.image_b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pth'
            torch.save(self.model, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
        actual = restored(self.image_a, self.image_b)
        torch.testing.assert_close(expected, actual)
        with self.assertRaisesRegex(RuntimeError, 'does not contain'):
            restored(self.image_a, self.image_b, return_aux=True)

    def test_full_model_checkpoint_round_trip(self):
        expected = self.model(
            self.image_a,
            self.image_b,
            return_aux=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(self.model, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
        restored.eval()
        actual = restored(
            self.image_a,
            self.image_b,
            return_aux=True,
        )
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name])


if __name__ == '__main__':
    unittest.main()
