import csv
import gc
import json
import time
from pathlib import Path

import torch

from guminet.data import make_loaders
from guminet.engine import GumiTrainer
from guminet.models import create_model
from guminet.utils import save_json, seed_everything


class GumiTuner:
    def __init__(self, config):
        self.config = dict(config)

    def run(self):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Install guminet[tuning] to use Optuna") from exc
        c = self.config
        root = Path(c.get("output_dir", "runs/tuning")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            study_name=c.get("study_name", "guminet"),
            storage=c.get("storage", f"sqlite:///{root / 'study.db'}"),
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=c.get("seed", 42)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        )

        def objective(trial):
            config = dict(c)
            config.update(
                lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", c.get("batch_sizes", [4, 8])),
                optimizer=trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
                input_size=trial.suggest_categorical("input_size", c.get("input_sizes", [128, 160])),
                scheduler=trial.suggest_categorical("scheduler", ["cosine", "step", "none"]),
                epochs=c.get("epochs_per_trial", 5),
                output_dir=str(root / f"trial_{trial.number}"),
            )
            seed_everything(config.get("seed", 42))
            loaders = make_loaders(
                config["dataset"],
                config["input_size"],
                config["batch_size"],
                config.get("num_workers", 0),
                config.get("seed", 42),
                include_test=False,
                **{k: config[k] for k in ("mean", "std") if k in config},
            )
            model = create_model(
                c.get("model", "guminet_nano"),
                num_classes=len(loaders["train"].dataset.classes),
                dropout=config["dropout"],
            )
            trainer = GumiTrainer(model, config, loaders["train"].dataset.class_to_idx)
            started = time.perf_counter()

            def callback(epoch, row):
                trial.set_user_attr("peak_process_ram_mb", row["peak_process_ram_mb"])
                trial.set_user_attr("cuda_peak_mb", row["cuda_peak_mb"])
                trial.set_user_attr("sampled_peak_training_ram_mb", row["sampled_peak_training_ram_mb"])
                trial.report(row["val_macro_f1"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                trainer.fit(loaders["train"], loaders["val"], callback)
                return trainer.best_metric
            finally:
                trial.set_user_attr("seconds", time.perf_counter() - started)
                del trainer, model, loaders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        study.optimize(objective, n_trials=c.get("trials", 5), n_jobs=1)
        save_json(
            root / "best_params.json",
            dict(params=study.best_params, macro_f1=study.best_value, trial=study.best_trial.number),
        )
        with (root / "trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["number", "state", "value", "params", "resources"])
            writer.writeheader()
            for trial in study.trials:
                writer.writerow(
                    dict(
                        number=trial.number,
                        state=trial.state.name,
                        value=trial.value,
                        params=json.dumps(trial.params),
                        resources=json.dumps(trial.user_attrs),
                    )
                )
        return study
