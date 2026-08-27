import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import get_ink_metrics


class InkMetricPortabilityTests(unittest.TestCase):
    def test_worker_uses_spawn_when_fork_is_unavailable(self):
        with mock.patch.object(
                get_ink_metrics.multiprocessing, 'get_all_start_methods',
                return_value=['spawn']), mock.patch.object(
                get_ink_metrics.multiprocessing, 'set_start_method') as setter:
            method = get_ink_metrics.configure_worker_start_method()

        self.assertEqual(method, 'spawn')
        setter.assert_called_once_with('spawn', force=True)

    def test_worker_prefers_fork_when_platform_provides_it(self):
        with mock.patch.object(
                get_ink_metrics.multiprocessing, 'get_all_start_methods',
                return_value=['fork', 'spawn']), mock.patch.object(
                get_ink_metrics.multiprocessing, 'set_start_method') as setter:
            method = get_ink_metrics.configure_worker_start_method()

        self.assertEqual(method, 'fork')
        setter.assert_called_once_with('fork', force=True)

    def test_public_trainer_bypasses_recursive_optional_module_scan(self):
        recursive = mock.Mock(side_effect=AssertionError('tree scan should not run'))
        finder = SimpleNamespace(recursive_find_python_class=recursive)
        trainer_class = object()
        module = SimpleNamespace(nnUNetTrainer_250epochs=trainer_class)
        with mock.patch.object(get_ink_metrics.importlib, 'import_module',
                               return_value=module) as importer:
            safe_find = get_ink_metrics.install_public_trainer_lookup(finder)
            actual = safe_find('/trainers', 'nnUNetTrainer_250epochs', 'root')

        self.assertIs(actual, trainer_class)
        importer.assert_called_once_with(
            'nnunetv2.training.nnUNetTrainer.variants.training_length.'
            'nnUNetTrainer_Xepochs')
        recursive.assert_not_called()

    def test_other_trainers_keep_nnunet_discovery(self):
        expected = object()
        recursive = mock.Mock(return_value=expected)
        finder = SimpleNamespace(recursive_find_python_class=recursive)
        safe_find = get_ink_metrics.install_public_trainer_lookup(finder)

        actual = safe_find('/trainers', 'CustomTrainer', 'root')

        self.assertIs(actual, expected)
        recursive.assert_called_once_with('/trainers', 'CustomTrainer', 'root')


if __name__ == '__main__':
    unittest.main()
