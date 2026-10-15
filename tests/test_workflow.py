import pytest
import torch

from guminet import GumiPredictor, GumiTrainer, create_model
from guminet.data import make_loaders
from guminet.engine import prepare_finetuning
from guminet.engine.checkpoint import load_checkpoint, model_from_checkpoint
from guminet.evaluation import evaluate
from guminet.utils import seed_everything


def train_small(dataset_root, tmp_path, epochs=1, **kwargs):
    seed_everything(9)
    loaders = make_loaders(dataset_root, 32, batch_size=2, seed=9)
    model = create_model("guminet_nano", 2)
    config = (
        dict(
            epochs=epochs,
            input_size=32,
            progress=False,
            output_dir=str(tmp_path / "run"),
            seed=9,
            accumulation_steps=2,
            scheduler="none",
        )
        | kwargs
    )
    trainer = GumiTrainer(model, config, loaders["train"].dataset.class_to_idx)
    return trainer, loaders


def test_train_evaluate_resume_infer(dataset_root, tmp_path):
    trainer, loaders = train_small(dataset_root, tmp_path)
    before = trainer.model.classifier.weight.detach().clone()
    history = trainer.fit(loaders["train"], loaders["val"])
    assert len(history) == 1 and history[0]["train_loss"] > 0
    assert not torch.equal(before, trainer.model.classifier.weight)
    for filename in ("best.pt", "last.pt", "history.json", "history.csv", "loss.png", "accuracy.png"):
        assert (tmp_path / "run" / filename).is_file()
    checkpoint = tmp_path / "run/last.pt"
    metadata = load_checkpoint(checkpoint)
    assert metadata["epoch"] == 0 and metadata["class_to_idx"] == {"a": 0, "b": 1}
    model, _ = model_from_checkpoint(checkpoint)
    result = evaluate(model, loaders["test"], "cpu", output_dir=tmp_path / "eval")
    assert 0 <= result["macro_f1"] <= 1 and sum(map(sum, result["confusion_matrix"])) == 6
    assert (tmp_path / "eval/per_class.csv").is_file()
    predictor = GumiPredictor.from_checkpoint(checkpoint, "cpu")
    results = predictor.predict_batch(
        [dataset_root / "test/a/0.png", dataset_root / "test/b/0.png"], batch_size=1
    )
    assert len(results) == 2
    assert abs(sum(results[0]["probabilities"].values()) - 1) < 1e-6
    resumed, loaders2 = train_small(dataset_root, tmp_path, epochs=2)
    resumed.resume(checkpoint)
    assert resumed.start_epoch == 1
    resumed.fit(loaders2["train"], loaders2["val"])
    assert len(resumed.history) == 2
    assert load_checkpoint(checkpoint)["epoch"] == 1


def test_resume_matches_uninterrupted(dataset_root, tmp_path):
    full, loaders = train_small(dataset_root, tmp_path / "full", epochs=2)
    full.fit(loaders["train"], loaders["val"])
    first, loaders = train_small(dataset_root, tmp_path / "staged", epochs=1)
    first.fit(loaders["train"], loaders["val"])
    resumed, loaders = train_small(dataset_root, tmp_path / "staged", epochs=2)
    resumed.resume(tmp_path / "staged/run/last.pt")
    resumed.fit(loaders["train"], loaders["val"])
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)


def test_finetune(dataset_root, tmp_path):
    trainer, loaders = train_small(dataset_root, tmp_path)
    trainer.fit(loaders["train"], loaders["val"])
    checkpoint = tmp_path / "run/best.pt"
    for mode in ("head", "partial", "full"):
        model, groups = prepare_finetuning(checkpoint, 3, mode)
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 3)
        if mode == "head":
            assert not any(p.requires_grad for p in model.features.parameters())
        if mode == "partial":
            assert any(p.requires_grad for p in model.features.parameters())
            assert not all(p.requires_grad for p in model.features.parameters())
        assert len(groups) == (1 if mode == "head" else 2)
    with pytest.raises(FileNotFoundError, match="no pretrained"):
        prepare_finetuning(tmp_path / "missing.pt", 2)


def test_checkpoint_rejects_metadata(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path)


def test_preprocessing_mismatch(dataset_root, tmp_path):
    trainer, loaders = train_small(dataset_root, tmp_path, input_size=128)
    with pytest.raises(ValueError, match="preprocessing"):
        trainer.fit(loaders["train"], loaders["val"])


def test_accumulation_short_window_matches_large_batch(tmp_path):
    import copy

    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(123)
    reference = create_model("guminet_nano", 2, dropout=0)
    # Isolate accumulation semantics from BatchNorm microbatch statistics.
    for module in reference.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = False
    data = TensorDataset(torch.randn(5, 3, 32, 32), torch.tensor([0, 1, 0, 1, 1]))
    models = []
    for batch_size, accumulation in ((2, 2), (4, 1)):
        model = copy.deepcopy(reference)
        trainer = GumiTrainer(
            model,
            dict(
                epochs=1,
                input_size=32,
                optimizer="sgd",
                lr=0.01,
                weight_decay=0,
                scheduler="none",
                accumulation_steps=accumulation,
                progress=False,
                output_dir=str(tmp_path / str(batch_size)),
            ),
        )
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        trainer.fit(loader, loader)
        models.append(model)
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=1e-4, atol=1e-7)


def test_head_only_keeps_backbone_fixed(dataset_root, tmp_path):
    trainer, loaders = train_small(dataset_root, tmp_path / "initial")
    trainer.fit(loaders["train"], loaders["val"])
    model, groups = prepare_finetuning(tmp_path / "initial/run/best.pt", 2, "head")
    before = {k: v.clone() for k, v in model.features.state_dict().items()}
    head = model.classifier.weight.detach().clone()
    tuned = GumiTrainer(
        model,
        dict(epochs=1, input_size=32, progress=False, output_dir=str(tmp_path / "head")),
        loaders["train"].dataset.class_to_idx,
        groups,
    )
    tuned.fit(loaders["train"], loaders["val"])
    assert all(torch.equal(v, before[k]) for k, v in model.features.state_dict().items())
    assert not torch.equal(head, model.classifier.weight)


def test_metrics_reserved_class_names():
    from guminet.evaluation import classification_metrics

    result = classification_metrics([0, 1], [0, 1], ["accuracy", "macro avg"])
    assert result["accuracy"] == 1
    assert result["per_class"]["accuracy"]["f1-score"] == 1
    assert result["per_class"]["macro avg"]["f1-score"] == 1
