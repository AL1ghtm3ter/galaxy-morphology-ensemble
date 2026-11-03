"""Small offline checks for the refactored pipeline, not benchmark results."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from galaxy_morphology import cli
from galaxy_morphology.checkpoints import load_checkpoint, save_checkpoint
from galaxy_morphology.data import GalaxyDataset, read_labels, split_labels
from galaxy_morphology.engine import train_epoch
from galaxy_morphology.inference import five_view_prediction, normalize_weights, predict_batch
from galaxy_morphology.metrics import regression_metrics
from galaxy_morphology.models import MultiTaskRegressor, build_model
from galaxy_morphology.tasks import LABEL_COLUMNS, NUM_TARGETS

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def tiny_model():
    return MultiTaskRegressor(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)), 8)


def create_dataset(root, count=10):
    images = root / "images_training_rev1"
    images.mkdir(parents=True)
    random = np.random.default_rng(7)
    rows = []
    for index in range(count):
        image_id = 1000 + index
        Image.fromarray(random.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(images / f"{image_id}.jpg")
        rows.append({"GalaxyID": image_id, **dict(zip(LABEL_COLUMNS, random.uniform(0, 1, NUM_TARGETS)))})
    pd.DataFrame(rows).to_csv(root / "training_solutions_rev1.csv", index=False)


class MetricsTests(unittest.TestCase):
    def test_rmse_and_task_slices(self):
        truth = np.zeros((4, NUM_TARGETS))
        result = regression_metrics(truth, truth + 0.25)
        self.assertAlmostEqual(result["global_rmse"], 0.25)
        self.assertEqual(len(result["task_rmse"]), 11)
        self.assertTrue(all(value == 0.25 for value in result["task_rmse"].values()))

    def test_root_accuracy_is_only_first_three_outputs(self):
        truth = np.zeros((2, NUM_TARGETS))
        truth[0, 0], truth[1, 1] = 1, 1
        predictions = truth.copy()
        predictions[0, :3] = [0, 1, 0]
        predictions[:, 3:] = 50
        self.assertEqual(regression_metrics(truth, predictions)["root_node_accuracy_percent"], 50)

    def test_reject_invalid_shapes_and_nonfinite_values(self):
        for truth, prediction in [(np.zeros((0, 37)), np.zeros((0, 37))),
                                  (np.zeros((1, 36)), np.zeros((1, 36))),
                                  (np.zeros((1, 37)), np.full((1, 37), np.nan))]:
            with self.assertRaises(ValueError):
                regression_metrics(truth, prediction)


class InferenceTests(unittest.TestCase):
    def test_exact_five_view_geometry(self):
        class Corner(nn.Module):
            def forward(self, values):
                return values[:, :1, 0, 0]
        image = torch.tensor([[[[0., 1.], [2., 3.]]]])
        self.assertAlmostEqual(five_view_prediction(Corner(), image).item(), 1.4, places=6)

    def test_weighted_ensemble_and_evaluation_mode(self):
        class Constant(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, images):
                return images.new_full((len(images), 37), self.value)
        models = [Constant(1), Constant(3)]
        result = predict_batch(models, torch.zeros(2, 3, 4, 4), [1, 3], tta=False)
        self.assertTrue(torch.allclose(result, torch.full((2, 37), 2.5)))
        self.assertTrue(all(not model.training for model in models))

    def test_invalid_weights(self):
        for values in [[0, 0], [-1, 2], [float("nan"), 1], [1]]:
            with self.assertRaises(ValueError):
                normalize_weights(values, 2)


class TrainingTests(unittest.TestCase):
    def test_partial_accumulation_matches_one_full_batch(self):
        torch.manual_seed(5)
        model = tiny_model()
        other = copy.deepcopy(model)
        images, targets = torch.rand(5, 3, 8, 8), torch.rand(5, 37)
        dataset = TensorDataset(images, targets)
        for network, batch_size, accumulation in [(model, 5, 1), (other, 2, 3)]:
            optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
            train_epoch(network, DataLoader(dataset, batch_size=batch_size), optimizer,
                        nn.MSELoss(), torch.device("cpu"), accumulation_steps=accumulation,
                        gradient_clip=100, amp=False)
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=1e-6))

    def test_nonfinite_loss_is_not_silently_ignored(self):
        model = tiny_model()
        loader = DataLoader(TensorDataset(torch.zeros(1, 3, 8, 8), torch.full((1, 37), float("nan"))))
        with self.assertRaises(FloatingPointError):
            train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.01),
                        nn.MSELoss(), torch.device("cpu"), amp=False)

    def test_task_heads_preserve_shape_and_gradients(self):
        model = tiny_model()
        result = model(torch.rand(2, 3, 8, 8))
        self.assertEqual(tuple(result.shape), (2, 37))
        result.sum().backward()
        self.assertTrue(all(getattr(model, f"task{i}").weight.grad is not None for i in range(1, 12)))


class DataAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        create_dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_repeatable_disjoint_split(self):
        frame = read_labels(self.root)
        training, validation = split_labels(frame)
        self.assertEqual((len(training), len(validation)), (8, 2))
        self.assertFalse(set(training.GalaxyID) & set(validation.GalaxyID))
        self.assertEqual(validation.GalaxyID.tolist(), split_labels(frame)[1].GalaxyID.tolist())

    def test_image_and_targets_have_expected_shape(self):
        dataset = GalaxyDataset(read_labels(self.root), self.root / "images_training_rev1")
        image, targets = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(tuple(targets.shape), (37,))

    def test_missing_image_fails_explicitly(self):
        (self.root / "images_training_rev1" / "1000.jpg").unlink()
        dataset = GalaxyDataset(read_labels(self.root), self.root / "images_training_rev1")
        with self.assertRaisesRegex(RuntimeError, "1000.jpg"):
            dataset[0]

    def test_duplicate_ids_and_invalid_labels_are_rejected(self):
        path = self.root / "training_solutions_rev1.csv"
        frame = pd.read_csv(path)
        frame.loc[1, "GalaxyID"] = frame.loc[0, "GalaxyID"]
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            read_labels(self.root)
        frame.loc[1, "GalaxyID"] = 1001
        frame.loc[0, LABEL_COLUMNS[0]] = -0.1
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            read_labels(self.root)

    def test_legacy_checkpoint_import(self):
        model = tiny_model()
        raw, wrapped = self.root / "raw.pth", self.root / "wrapped.pt"
        torch.save(model.state_dict(), raw)
        args = argparse.Namespace(weights_file=str(raw), output=str(wrapped),
                                  model="vit", data_dir=str(self.root), seed=42)
        with patch("galaxy_morphology.models.build_model", return_value=tiny_model()):
            cli.import_legacy(args)
        metadata = torch.load(wrapped, weights_only=True)
        self.assertEqual(metadata["architecture"], "vit")
        self.assertEqual(len(metadata["validation_ids"]), 2)

    def test_ensemble_rejects_different_validation_splits(self):
        paths = [self.root / "a.pt", self.root / "b.pt"]
        for path, ids in zip(paths, [[1001, 1002], [1002, 1003]]):
            save_checkpoint(path, tiny_model(), "vit", ids)
        args = argparse.Namespace(checkpoints=[str(path) for path in paths], device="cpu",
                                  weights=None, data_dir=str(self.root), batch_size=2,
                                  no_tta=False, no_amp=True, output=str(self.root / "result.json"))
        with patch("galaxy_morphology.checkpoints.build_model", side_effect=lambda *a, **kw: tiny_model()):
            with self.assertRaisesRegex(ValueError, "same ordered validation IDs"):
                cli.evaluate(args)

    def test_checkpoint_round_trip_avoids_weight_download(self):
        model = tiny_model().eval()
        path = self.root / "model.pt"
        save_checkpoint(path, model, "efficientnet", [1001, 1003])
        with patch("galaxy_morphology.checkpoints.build_model", return_value=tiny_model()) as builder:
            loaded, metadata = load_checkpoint(path, torch.device("cpu"))
        builder.assert_called_once_with("efficientnet", pretrained=False)
        sample = torch.rand(2, 3, 8, 8)
        self.assertTrue(torch.equal(model(sample), loaded(sample)))
        self.assertEqual(metadata["validation_ids"], [1001, 1003])

    def test_train_evaluate_predict_integration(self):
        output = self.root / "run"
        args = argparse.Namespace(config=str(ROOT / "configs" / "baseline.json"), model="efficientnet",
                                  data_dir=str(self.root), output_dir=str(output), device="cpu", epochs=1,
                                  batch_size=3, limit=None, no_pretrained=True, no_amp=True)
        with patch("galaxy_morphology.models.build_model", side_effect=lambda *a, **kw: tiny_model()):
            trained = cli.train(args)
        checkpoint = output / "best.pt"
        self.assertTrue(checkpoint.is_file())
        evaluation_args = argparse.Namespace(checkpoints=[str(checkpoint)], weights=None, device="cpu",
                                            data_dir=str(self.root), batch_size=2, no_tta=False,
                                            no_amp=True, output=str(output / "evaluation.json"))
        with patch("galaxy_morphology.checkpoints.build_model", side_effect=lambda *a, **kw: tiny_model()):
            evaluated = cli.evaluate(evaluation_args)
            prediction_args = argparse.Namespace(checkpoints=[str(checkpoint)], weights=None, device="cpu",
                                                 image=str(self.root / "images_training_rev1" / "1000.jpg"),
                                                 no_tta=False, output=str(output / "prediction.json"))
            prediction = cli.predict(prediction_args)
        self.assertEqual(evaluated["samples"], 2)
        self.assertEqual(len(prediction["targets"]), 37)
        self.assertTrue(np.isfinite(trained["best_validation_rmse"]))
        self.assertEqual(json.loads((output / "run.json").read_text())["epochs"], 1)


class BackboneTests(unittest.TestCase):
    def test_all_three_real_backbones_without_downloads(self):
        import gc
        for architecture in ("vit", "swin", "efficientnet"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture, pretrained=False).eval()
                with torch.inference_mode():
                    result = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(result.shape), (1, 37))
                self.assertTrue(torch.isfinite(result).all())
                del model, result
                gc.collect()


if __name__ == "__main__":
    unittest.main()

