import json
import subprocess
import sys
import unittest
from pathlib import Path

from click.testing import CliRunner


SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import render_ink


class RenderInkPathTests(unittest.TestCase):
    def test_local_render_command_is_unchanged_by_default(self):
        command = render_ink.build_vc_render_command(
            'vc_render_tifxyz', '/seg/w001', '/out',
            volume='/data/ink.zarr', scale=0.25, group_idx=1, num_slices=5,
        )

        self.assertEqual(command, [
            'vc_render_tifxyz', '--segmentation', '/seg/w001',
            '--scale', '0.25', '--group-idx', '1',
            '--volume', '/data/ink.zarr', '--tif-output', '/out',
            '--num-slices', '5',
        ])

    def test_remote_render_options_match_vc_render_types(self):
        command = render_ink.build_vc_render_command(
            'vc_render_tifxyz', '/seg/w059', '/out',
            volume='/tmp/unused', scale=1.0, group_idx=2, num_slices=5,
            remote_url='https://example.test/ink.zarr', scale_segmentation=0.25,
            slice_step=0.5, cache_gb=4, prefetch_remote=True,
            crop_x=1198, crop_y=302, crop_width=2048, crop_height=768,
        )

        self.assertEqual(command, [
            'vc_render_tifxyz', '--segmentation', '/seg/w059',
            '--scale', '1.0', '--group-idx', '2',
            '--volume', '/tmp/unused', '--tif-output', '/out',
            '--num-slices', '5',
            '--remote-url', 'https://example.test/ink.zarr',
            '--scale-segmentation', '0.25', '--slice-step', '0.5',
            '--cache-gb', '4', '--prefetch-remote',
            '--crop-x', '1198', '--crop-y', '302',
            '--crop-width', '2048', '--crop-height', '768',
        ])

    def test_remote_option_click_types_match_vc_render_types(self):
        options = {parameter.name: parameter for parameter in render_ink.main.params}

        self.assertEqual(options['scale_segmentation'].type.name, 'float')
        self.assertEqual(options['slice_step'].type.name, 'float')
        self.assertEqual(options['cache_gb'].type.name, 'integer range')
        self.assertEqual(options['scale_segmentation'].type_cast_value(None, '0.25'), 0.25)
        self.assertEqual(options['slice_step'].type_cast_value(None, '0.5'), 0.5)
        self.assertEqual(options['cache_gb'].type_cast_value(None, '4'), 4)
        with self.assertRaises(render_ink.click.BadParameter):
            options['cache_gb'].type_cast_value(None, '4.5')
        with self.assertRaises(render_ink.click.BadParameter):
            options['cache_gb'].type_cast_value(None, '-1')

    def test_partial_crop_is_rejected(self):
        with self.assertRaisesRegex(render_ink.click.UsageError, 'must be given together'):
            render_ink.build_vc_render_command(
                'vc_render_tifxyz', '/seg/w059', '/out',
                volume='/tmp/unused', scale=1.0, group_idx=2, num_slices=5,
                crop_x=1,
            )

    def test_default_lasagna_dir_is_sibling_of_spiral_fitting(self):
        script = Path("/checkout/spiral-fitting/render_ink.py")

        actual = Path(render_ink.default_lasagna_dir(script))

        self.assertEqual(actual, Path("/checkout/lasagna"))

    def test_failed_full_scroll_flatten_fails_when_no_strips_are_rendered(self):
        with CliRunner().isolated_filesystem():
            meshes_dir = Path("meshes")
            mesh = meshes_dir / "w001_spliced"
            mesh.mkdir(parents=True)
            (mesh / "meta.json").write_text(json.dumps({"format": "tifxyz"}))

            original_read = render_ink.read_step_and_voxel
            original_build = render_ink.build_full_concat
            original_flatten = render_ink.lasagna_flatten
            try:
                render_ink.read_step_and_voxel = lambda _path: (1, 1.0)
                render_ink.build_full_concat = lambda *_args: (
                    "w001-001", "meshes/concat/w001-001", 10)

                def fail_flatten(*_args):
                    raise subprocess.CalledProcessError(1, ["lasagna"])

                render_ink.lasagna_flatten = fail_flatten
                result = CliRunner().invoke(render_ink.main, [
                    str(meshes_dir), "--volume", "ink.zarr",
                ])
            finally:
                render_ink.read_step_and_voxel = original_read
                render_ink.build_full_concat = original_build
                render_ink.lasagna_flatten = original_flatten

        self.assertEqual(result.exit_code, 1)
        self.assertIn("render produced no ink strip images", result.output)


if __name__ == "__main__":
    unittest.main()
