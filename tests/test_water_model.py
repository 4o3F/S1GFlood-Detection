from __future__ import annotations

import inspect
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from timm.models.helpers import adapt_input_conv

from swin_transformer import swin
from water_seg.model import DecoderBlock, SwinTinyEncoder, SwinTinyUNet


def _clipped_db_vv(batch=1, height=64, width=64):
    return torch.empty(batch, 1, height, width).uniform_(-32.0, 0.0)


class SwinTinyEncoderInitTest(unittest.TestCase):
    def test_is_local_swin_subclass(self):
        self.assertTrue(issubclass(SwinTinyEncoder, swin))
        self.assertIsNot(SwinTinyEncoder.init_weights, swin.init_weights)

    def test_init_weights_does_not_use_legacy_pretrained(self):
        source = inspect.getsource(SwinTinyEncoder.init_weights)
        self.assertNotIn('PRETRAINED', source)
        self.assertNotIn('torch.load', source)

    def test_construction_does_not_call_torch_load(self):
        with patch('torch.load') as torch_load, patch(
            'swin_transformer.torch.load'
        ) as module_load:
            encoder = SwinTinyEncoder()
            torch_load.assert_not_called()
            module_load.assert_not_called()
        self.assertEqual(list(encoder.num_features), [96, 192, 384, 768])

    def test_patch_embed_uses_single_channel_stem(self):
        source = inspect.getsource(SwinTinyEncoder.__init__)
        self.assertIn('in_chans=1', source)
        self.assertNotIn('in_chans=3', source)
        encoder = SwinTinyEncoder()
        self.assertEqual(encoder.patch_embed.in_chans, 1)
        self.assertEqual(encoder.patch_embed.proj.in_channels, 1)
        self.assertEqual(tuple(encoder.patch_embed.proj.weight.shape), (96, 1, 4, 4))


class SwinTinyUNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        with patch('torch.load') as torch_load:
            cls.model = SwinTinyUNet(imagenet_pretrained=False).eval()
            torch_load.assert_not_called()

    def test_encoder_is_exposed_for_optimizer_grouping(self):
        self.assertIsInstance(self.model.encoder, SwinTinyEncoder)
        self.assertTrue(
            any(name.startswith('encoder.') for name, _ in self.model.named_parameters())
        )

    def test_encoder_patch_embed_in_chans_is_one(self):
        self.assertEqual(self.model.encoder.patch_embed.in_chans, 1)

    def test_encoder_stage_channels(self):
        features = self.model.encoder(torch.rand(1, 1, 64, 64))
        self.assertEqual(len(features), 4)
        for feature, channels, stride in zip(
            features,
            (96, 192, 384, 768),
            (4, 8, 16, 32),
        ):
            self.assertEqual(tuple(feature.shape), (1, channels, 64 // stride, 64 // stride))

    def test_forward_returns_full_resolution_logits(self):
        image = _clipped_db_vv()
        with torch.no_grad():
            logits = self.model(image)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))

    def test_rejects_three_channel_input(self):
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 3, 64, 64))

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 64, 64))
        with self.assertRaises(ValueError):
            self.model(torch.rand(64, 64))
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 2, 64, 64))

    def test_decoder_outputs_are_512_256_128_64(self):
        self.assertEqual(self.model.dec1.out_channels, 512)
        self.assertEqual(self.model.dec2.out_channels, 256)
        self.assertEqual(self.model.dec3.out_channels, 128)
        self.assertEqual(self.model.dec4.out_channels, 64)
        self.assertIsInstance(self.model.dec1, DecoderBlock)
        self.assertEqual(self.model.dec4.skip_channels, 0)

        channels = []
        hooks = [
            block.register_forward_hook(
                lambda module, inputs, output, bucket=channels: bucket.append(
                    output.shape[1]
                )
            )
            for block in (
                self.model.dec1,
                self.model.dec2,
                self.model.dec3,
                self.model.dec4,
            )
        ]
        try:
            with torch.no_grad():
                self.model(_clipped_db_vv())
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(channels, [512, 256, 128, 64])

    def test_has_vv_buffers_and_no_imagenet_or_repeat_behavior(self):
        self.assertEqual(tuple(self.model.vv_mean.shape), (1, 1, 1, 1))
        self.assertEqual(tuple(self.model.vv_std.shape), (1, 1, 1, 1))
        torch.testing.assert_close(self.model.vv_mean, torch.zeros(1, 1, 1, 1))
        torch.testing.assert_close(self.model.vv_std, torch.ones(1, 1, 1, 1))
        self.assertFalse(hasattr(self.model, 'imagenet_mean'))
        self.assertFalse(hasattr(self.model, 'imagenet_std'))
        source = inspect.getsource(SwinTinyUNet)
        self.assertNotIn('imagenet_mean', source)
        self.assertNotIn('imagenet_std', source)
        self.assertNotIn('/255', source.replace(' ', ''))
        self.assertNotIn('repeat(', source)

    def test_set_vv_normalization_updates_and_rejects_invalid_values(self):
        model = SwinTinyUNet(imagenet_pretrained=False)
        returned = model.set_vv_normalization(-12.5, 4.0)
        self.assertIs(returned, model)
        torch.testing.assert_close(
            model.vv_mean, torch.tensor(-12.5).view(1, 1, 1, 1)
        )
        torch.testing.assert_close(
            model.vv_std, torch.tensor(4.0).view(1, 1, 1, 1)
        )
        for mean, std in (
            (0.0, 0.0),
            (0.0, -1.0),
            (float('nan'), 1.0),
            (float('inf'), 1.0),
            (0.0, float('nan')),
            (0.0, float('inf')),
            (-math.inf, 1.0),
            (0.0, -math.inf),
        ):
            with self.assertRaises(ValueError):
                model.set_vv_normalization(mean, std)

    def test_vv_normalization_changes_prepared_and_forward_values(self):
        model = SwinTinyUNet(imagenet_pretrained=False).eval()
        image = torch.linspace(-32.0, 0.0, 64 * 64, dtype=torch.float32).view(
            1, 1, 64, 64
        )
        default_prepared = model._prepare_input(image)
        torch.testing.assert_close(default_prepared, image)
        with torch.no_grad():
            default_logits = model(image)

        model.set_vv_normalization(-16.0, 8.0)
        prepared = model._prepare_input(image)
        torch.testing.assert_close(prepared, (image - (-16.0)) / 8.0)
        self.assertEqual(prepared.dtype, model.vv_mean.dtype)
        self.assertEqual(prepared.device, model.vv_mean.device)
        with torch.no_grad():
            normalized_logits = model(image)
            again = model(image)
        self.assertFalse(torch.allclose(normalized_logits, default_logits))
        torch.testing.assert_close(again, normalized_logits)

    def test_state_dict_round_trip_preserves_vv_stats_and_logits(self):
        model = SwinTinyUNet(imagenet_pretrained=False).eval()
        model.set_vv_normalization(-15.0, 8.0)
        image = torch.linspace(-32.0, 0.0, 64 * 64, dtype=torch.float32).view(
            1, 1, 64, 64
        )
        with torch.no_grad():
            original = model(image)
        clone = SwinTinyUNet(imagenet_pretrained=False)
        clone.load_state_dict(model.state_dict())
        clone.eval()
        torch.testing.assert_close(clone.vv_mean, model.vv_mean)
        torch.testing.assert_close(clone.vv_std, model.vv_std)
        with torch.no_grad():
            restored = clone(image)
        torch.testing.assert_close(restored, original)


class ImageNetPretrainedLoadTest(unittest.TestCase):
    def test_default_construction_does_not_create_timm_model(self):
        with patch('timm.create_model') as create_model:
            SwinTinyUNet(imagenet_pretrained=False)
            create_model.assert_not_called()

    def test_loads_matching_encoder_keys_from_timm_classification_model(self):
        import timm

        source = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            in_chans=3,
        )
        with patch('timm.create_model', return_value=source) as create_model:
            model = SwinTinyUNet(imagenet_pretrained=True)
        create_model.assert_called_once_with(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            in_chans=3,
        )
        kwargs = create_model.call_args.kwargs
        self.assertNotIn('features_only', kwargs)

        source_state = source.state_dict()
        loaded = model.encoder.state_dict()
        adapted_stem = adapt_input_conv(1, source_state['patch_embed.proj.weight'])
        self.assertEqual(tuple(loaded['patch_embed.proj.weight'].shape), (96, 1, 4, 4))
        torch.testing.assert_close(
            loaded['patch_embed.proj.weight'],
            adapted_stem,
        )
        torch.testing.assert_close(
            loaded['layers.0.blocks.0.attn.qkv.weight'],
            source_state['layers.0.blocks.0.attn.qkv.weight'],
        )
        self.assertIn('norm0.weight', loaded)
        torch.testing.assert_close(
            loaded['norm3.weight'],
            source_state['norm.weight'],
        )
        torch.testing.assert_close(
            loaded['norm3.bias'],
            source_state['norm.bias'],
        )
        self.assertNotIn('head.weight', loaded)
        self.assertNotIn('head.bias', loaded)

    def test_fails_when_too_few_tensors_match(self):
        class TinySource(nn.Module):
            def state_dict(self, *args, **kwargs):
                return {
                    'head.weight': torch.randn(1000, 768),
                    'head.bias': torch.randn(1000),
                    'norm.weight': torch.randn(768),
                    'norm.bias': torch.randn(768),
                    'layers.0.blocks.1.attn_mask': torch.zeros(64, 49, 49),
                }

        with patch('timm.create_model', return_value=TinySource()):
            with self.assertRaises(RuntimeError) as raised:
                SwinTinyUNet(imagenet_pretrained=True)
        self.assertIn('matched 2 tensors', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
