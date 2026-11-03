"""Train, evaluate and predict from the repository root."""
import argparse
import json
import os
import random
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_device(value):
    import torch
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Choose --device cpu.")
    return torch.device(value)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from .checkpoints import save_checkpoint
    from .data import GalaxyDataset, read_labels, split_labels
    from .engine import evaluate_models, train_epoch
    from .models import build_model

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("image_size") != 224:
        raise ValueError("The supplied backbones and preprocessing expect 224-pixel inputs.")
    settings = config["models"][args.model]
    seed_everything(config["seed"])
    device = resolve_device(args.device)
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    config = {**config, "epochs": epochs, "batch_size": batch_size}
    frame = read_labels(args.data_dir, args.limit)
    training, validation = split_labels(frame, config["seed"], config["validation_fraction"])
    image_dir = Path(args.data_dir) / "images_training_rev1"
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(GalaxyDataset(training, image_dir, True), batch_size=batch_size,
                              shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(GalaxyDataset(validation, image_dir), batch_size=batch_size,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=settings["minimum_learning_rate"])
    criterion = (torch.nn.HuberLoss(delta=settings["huber_delta"])
                 if settings["loss"] == "huber" else torch.nn.MSELoss())
    output = Path(args.output_dir)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device,
                           config["accumulation_steps"], config["gradient_clip"], not args.no_amp)
        metrics = evaluate_models([model], val_loader, device, tta=False, amp=not args.no_amp)
        scheduler.step()
        row = {"epoch": epoch, "training_loss": loss, "validation_rmse": metrics["global_rmse"]}
        history.append(row)
        write_json(output / "history.json", history)
        if metrics["global_rmse"] < best:
            best = metrics["global_rmse"]
            save_checkpoint(output / "best.pt", model, args.model, validation["GalaxyID"].tolist(),
                            epoch=epoch, config=config, validation_metrics=metrics,
                            training_samples=len(training), validation_samples=len(validation))
        print(json.dumps(row), flush=True)
    write_json(output / "run.json", {"architecture": args.model, "device": str(device),
               "pretrained": not args.no_pretrained, "epochs": epochs, "batch_size": batch_size,
               "sample_limit": args.limit, "best_validation_rmse": best, "config": config})
    return {"checkpoint": str(output / "best.pt"), "best_validation_rmse": best}


def load_models(args):
    from .checkpoints import load_checkpoint
    from .inference import normalize_weights
    device = resolve_device(args.device)
    weights = normalize_weights(args.weights, len(args.checkpoints))
    pairs = [load_checkpoint(path, device) for path in args.checkpoints]
    return device, [pair[0] for pair in pairs], [pair[1] for pair in pairs], weights


def import_legacy(args):
    import torch
    from .checkpoints import save_checkpoint
    from .data import read_labels, split_labels
    from .models import build_model
    state = torch.load(args.weights_file, map_location="cpu", weights_only=True)
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(state, strict=True)
    _, validation = split_labels(read_labels(args.data_dir), seed=args.seed)
    save_checkpoint(args.output, model, args.model, validation["GalaxyID"].tolist(),
                    imported_from=Path(args.weights_file).name, split_seed=args.seed,
                    source="Legacy raw coursework state dictionary; metrics require evaluation.")
    return {"checkpoint": args.output, "validation_samples": len(validation)}


def evaluate(args):
    from torch.utils.data import DataLoader
    from .data import GalaxyDataset, read_labels
    from .engine import evaluate_models
    device, models, checkpoints, weights = load_models(args)
    ids = checkpoints[0]["validation_ids"]
    if any(item["validation_ids"] != ids for item in checkpoints[1:]):
        raise ValueError("All ensemble checkpoints must use the same ordered validation IDs.")
    frame = read_labels(args.data_dir).set_index("GalaxyID")
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"The CSV is missing {len(missing)} checkpoint validation IDs.")
    validation = frame.loc[ids].reset_index()
    loader = DataLoader(GalaxyDataset(validation, Path(args.data_dir) / "images_training_rev1"),
                        batch_size=args.batch_size, num_workers=0)
    result = evaluate_models(models, loader, device, weights, not args.no_tta, not args.no_amp)
    result["evaluation_scope"] = "Checkpoint validation split; not an independent test set."
    result["architectures"] = [item["architecture"] for item in checkpoints]
    result["weights"] = weights
    result["tta_views"] = 1 if args.no_tta else 5
    write_json(args.output, result)
    return result


def predict(args):
    from PIL import Image
    from .data import image_transform
    from .inference import predict_batch
    from .tasks import LABEL_COLUMNS
    device, models, checkpoints, weights = load_models(args)
    with Image.open(args.image) as image:
        tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    values = predict_batch(models, tensor, weights, not args.no_tta)[0].float().cpu().tolist()
    result = {"targets": dict(zip(LABEL_COLUMNS, values)),
              "output_type": "Unconstrained vote-fraction regression outputs.",
              "architectures": [item["architecture"] for item in checkpoints]}
    write_json(args.output, result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Galaxy Zoo morphology training and ensemble evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train", help="Train one backbone on a seeded local split")
    fitting.add_argument("--model", choices=["vit", "swin", "efficientnet"], required=True)
    fitting.add_argument("--config", default="configs/baseline.json")
    fitting.add_argument("--output-dir", required=True)
    fitting.add_argument("--epochs", type=int)
    fitting.add_argument("--batch-size", type=int)
    fitting.add_argument("--limit", type=int, help="First N CSV rows for smoke tests only")
    fitting.add_argument("--no-pretrained", action="store_true", help="Avoid weight downloads; useful for offline smoke tests")
    fitting.set_defaults(function=train)
    checking = commands.add_parser("evaluate", help="Evaluate one model or a weighted ensemble")
    checking.add_argument("--output", default="runs/evaluation.json")
    checking.add_argument("--data-dir", default=os.environ.get("GALAXY_DATA_DIR", "data"))
    checking.add_argument("--batch-size", type=int, default=8)
    checking.set_defaults(function=evaluate)
    prediction = commands.add_parser("predict", help="Predict 37 targets for one image")
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--output", default="runs/prediction.json")
    prediction.set_defaults(function=predict)
    legacy = commands.add_parser("import-legacy", help="Wrap an original raw coursework state dictionary")
    legacy.add_argument("--weights-file", required=True)
    legacy.add_argument("--model", choices=["vit", "swin", "efficientnet"], required=True)
    legacy.add_argument("--data-dir", default=os.environ.get("GALAXY_DATA_DIR", "data"))
    legacy.add_argument("--seed", type=int, default=42)
    legacy.add_argument("--output", required=True)
    legacy.set_defaults(function=import_legacy)
    for command in (checking, prediction):
        command.add_argument("--checkpoints", nargs="+", required=True)
        command.add_argument("--weights", nargs="+", type=float)
        command.add_argument("--no-tta", action="store_true")
    fitting.add_argument("--data-dir", default=os.environ.get("GALAXY_DATA_DIR", "data"))
    for command in (fitting, checking):
        command.add_argument("--no-amp", action="store_true")
    for command in (fitting, checking, prediction):
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) is not None and getattr(args, "batch_size", 1) < 1:
        parser.error("Batch size must be positive.")
    try:
        result = args.function(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))

