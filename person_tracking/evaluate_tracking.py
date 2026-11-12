"""Evaluate a complete labelled video without a desktop or camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import cv2
import torch

from person_tracking import CACHE, PersonTracking, select_device
from tracking_evaluation import SequenceEvaluator, Target, load_ground_truth


def evaluate_video(video, ground_truth, tracking, evaluator, experiment=None):
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("평가 영상을 열 수 없습니다.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("영상의 FPS를 읽을 수 없습니다. 유효한 프레임 속도의 영상을 사용하세요.")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        started = time.monotonic()
        completed = started
        if experiment is not None:
            experiment.start_pipeline(started)
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            observations, assignments = tracking.process(frame, frame_number / fps)
            tracks = [Target(item.person_id, tuple(observation.box))
                      for observation, item in zip(observations, assignments) if item.person_id is not None]
            evaluator.add_frame(tracking.last_detection_boxes, tracks)
            frame_number += 1
            completed = time.monotonic()
            if experiment is not None:
                experiment.complete_pipeline_frame(completed)
        if expected_frames > 0 and frame_number != expected_frames:
            raise ValueError(f"영상 디코딩이 중단되었습니다: 예상 {expected_frames}, 처리 {frame_number}프레임")
        report = evaluator.result()
        elapsed = completed - started
        report["metrics"].update(actual_fps=frame_number / elapsed,
                                 pipeline_frames=frame_number, pipeline_seconds=elapsed)
        sources = {}
        for name, path in (("video", video), ("ground_truth", ground_truth)):
            with path.open("rb") as stream:
                sources[f"{name}_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["protocol"].update(sources)
        report["protocol"].update(video_fps=fps, trackeval_version=version("trackeval"),
                                  pipeline_scope="offline decode + tracking + metric input accumulation + periodic logging; no GUI")
        return report
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description="정답 라벨이 있는 영상의 탐지·ID 추적 성능 평가")
    parser.add_argument("--video", type=Path, required=True, help="평가 영상. 모든 프레임을 순서대로 처리")
    parser.add_argument("--ground-truth", type=Path, required=True, help="사람 전용 MOT 형식 정답 파일(9열)")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"), help="집계 결과 JSON")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--reid-interval", type=float, default=0.2)
    parser.add_argument("--reid-batch", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--reid-gap-frames", type=int, default=1, help="재입장으로 셀 최소 정답 부재 프레임 수")
    parser.add_argument("--no-aim", action="store_true", help="AIM-LTCC 비활성화 비교 실험")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name", default="video-evaluation")
    parser.add_argument("--log-models", action="store_true")
    args = parser.parse_args()
    if (args.size < 128 or not math.isfinite(args.reid_interval) or args.reid_interval <= 0
            or args.reid_batch < 1 or not math.isfinite(args.iou_threshold)
            or not 0 < args.iou_threshold <= 1 or args.reid_gap_frames < 1):
        parser.error("크기는 128 이상, 간격·배치는 양수, IoU는 (0,1], 부재 프레임은 1 이상이어야 합니다.")
    if not args.mlflow and (args.tracking_uri or args.experiment_name or args.log_models):
        parser.error("MLflow 관련 옵션은 --mlflow와 함께 사용하세요.")
    if args.output.resolve() in {args.video.resolve(), args.ground_truth.resolve()}:
        parser.error("출력 경로는 입력 영상·정답 파일과 달라야 합니다.")
    try:
        # Validate inputs/dependencies before loading the large model checkpoints.
        import trackeval.metrics
        ground_truth = load_ground_truth(args.ground_truth)
        if not args.video.is_file():
            raise ValueError("평가 영상 파일이 없습니다.")
        evaluator = SequenceEvaluator(ground_truth, args.iou_threshold, args.reid_gap_frames)
        device = select_device(args.device)
        if device.type == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 6))
        experiment = None
        if args.mlflow:
            from experiment_tracking import TrackingExperiment
            experiment = TrackingExperiment(args.tracking_uri, args.experiment_name, args.run_name)
        with experiment if experiment is not None else nullcontext():
            tracking = PersonTracking(device, args.size, args.reid_interval, args.reid_batch, experiment, use_aim=not args.no_aim)
            if experiment is not None:
                experiment.client.set_tag(experiment.run_id, "task", "tracking-evaluation")
                experiment.configure(
                    {"device": str(device), "image_size": args.size, "reid_interval": args.reid_interval,
                     "reid_batch": args.reid_batch, "aim_enabled": not args.no_aim, "evaluation.iou_threshold": args.iou_threshold,
                     "evaluation.reid_gap_frames": args.reid_gap_frames, "pipeline_mode": "offline-video",
                     "detector_confidence": 0.10, "max_detections": 100,
                     "gmc_method": "orb", "track_buffer": 30},
                    tracking.identities.config, tracking.checkpoints, CACHE, args.log_models)
            report = evaluate_video(args.video, args.ground_truth, tracking, evaluator, experiment)
            report["protocol"]["aim_enabled"] = not args.no_aim
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            if experiment is not None:
                experiment.log_evaluation(report)
            print(json.dumps(report["metrics"], indent=2, allow_nan=False))
            print(f"평가 결과: {args.output}")
    except ImportError as error:
        print(f"평가 의존성을 설치하세요: pip install -r person_tracking/requirements-evaluation.txt ({error})", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        print(f"평가 오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
