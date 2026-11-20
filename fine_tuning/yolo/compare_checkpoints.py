"""Compare checkpoints on the same collapsed person-box detection task."""

from functools import partial
from hashlib import file_digest
import os
from pathlib import Path
from uuid import uuid4


def compare_checkpoints(args, config, data, prepared, baseline, fine_tuned, image_count):
    try:
        from mlflow import MlflowClient
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        raise ValueError("MLflow를 설치하세요: pip install -r fine_tuning/requirements-mlflow.txt") from error

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import nms

    class PersonValidator(DetectionValidator):
        def __init__(self, *, prediction_classes, **kwargs):
            super().__init__(**kwargs)
            self.prediction_classes = prediction_classes

        def postprocess(self, preds):
            # Filter predictions independently so dataset loading retains every GT class.
            outputs = nms.non_max_suppression(
                preds, self.args.conf, self.args.iou, classes=self.prediction_classes,
                multi_label=True, agnostic=True, max_det=self.args.max_det, end2end=self.end2end,
            )
            return [{"bboxes": boxes[:, :4], "conf": boxes[:, 4], "cls": boxes[:, 5],
                     "extra": boxes[:, 6:]} for boxes in outputs]

    checkpoint_pairs = [("before", baseline), ("after", fine_tuned)]
    models = [YOLO(str(path), task="detect") for _, path in checkpoint_pairs]
    if models[0].names.get(0) != "person":
        raise ValueError("학습 전 가중치는 0번 클래스가 person인 YOLO 가중치여야 합니다.")
    names = dict(enumerate(data["names"])) if isinstance(data["names"], list) else data["names"]
    if models[1].names != names:
        raise ValueError("학습 후 가중치의 클래스 이름과 ID가 데이터 YAML의 names와 일치해야 합니다.")

    default_uri = "sqlite:///" + str(Path(__file__).resolve().parents[2] / "mlflow.db")
    client = MlflowClient(tracking_uri=args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_uri)
    name = args.experiment_name or "classroom-yolo-detection"
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    comparison_id = uuid4().hex
    validation = {
        "split": args.split, "device": args.device, "imgsz": config.get("imgsz", 768),
        "batch": config.get("batch", 4), "workers": config.get("workers", 0),
        "single_cls": True, "classes": None, "conf": 0.001, "iou": 0.7,
        "max_det": 300, "fraction": 1.0, "augment": False, "nms": False,
    }
    for (stage, path), model in zip(checkpoint_pairs, models):
        run = client.create_run(experiment_id, tags={
            "mlflow.runName": f"{stage}-{args.split}-{comparison_id[:8]}",
            "comparison_id": comparison_id, "stage": stage,
            "task": "class-agnostic-person-detection",
        })
        run_id = run.info.run_id
        try:
            with path.open("rb") as stream:
                weight_sha256 = file_digest(stream, "sha256").hexdigest()
            for key, value in {
                **validation, "images": image_count, "dataset": prepared.name,
                "weights": path.name, "weights_sha256": weight_sha256,
                "prediction_classes": "0: person" if stage == "before" else "all dataset classes",
            }.items():
                client.log_param(run_id, key, value)
            validator = partial(PersonValidator, prediction_classes=[0] if stage == "before" else None)
            result = model.val(
                validator=validator, data=str(prepared), **validation,
                project=config["project"], name=f"compare-{comparison_id[:8]}-{stage}",
                save=False, save_json=False, save_txt=False, plots=False,
            )
            for key, value in {
                "precision": result.box.mp, "recall": result.box.mr,
                "map50": result.box.map50, "map50_95": result.box.map,
                "inference_ms_per_image": result.speed["inference"],
            }.items():
                client.log_metric(run_id, key, float(value))
        except KeyboardInterrupt:
            client.set_terminated(run_id, "KILLED")
            raise
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
        else:
            client.set_terminated(run_id)
            print(f"MLflow {stage}: {run_id} (comparison_id={comparison_id})", flush=True)
