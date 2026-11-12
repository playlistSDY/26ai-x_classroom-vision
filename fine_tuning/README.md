# Fine-tuning 작업 공간

강의실 데이터로 모델을 학습하고 결과를 비교하는 공간입니다.
YOLO는 학습·탐지 검증 실행기를 제공하며, AIM과 BPBreID는 데이터 준비 및 학습 코드 연결을 위한 공간입니다.
학습은 자동으로 시작하지 않습니다. 현재 실제 학습 데이터는 포함하지 않습니다.

```text
fine_tuning/
  configs/                 # 버전 관리할 데이터 경로·학습 설정
  yolo/run.py              # YOLO 학습·검증·사전 검사
  aim/                     # AIM-LTCC 학습 준비 안내
  bpbreid/                 # BPBreID 학습 준비 안내
  data/                    # 로컬 데이터 (Git 제외)
    yolo/images/{train,val,test}/
    yolo/labels/{train,val,test}/
    reid/                  # 인물 ID별 크롭·메타데이터 준비
  checkpoints/             # 로컬 사전학습 가중치 보관 (Git 제외)
  runs/                    # 학습 결과·평가·해석된 데이터 설정 (Git 제외)
```

## 환경

프로젝트 루트에서 실행합니다. 기존 `.venv`를 사용하거나 별도 학습 환경을 만들 수 있습니다.
UI 의존성 없이 학습할 수 있도록 requirements를 분리했습니다.

```sh
.venv/bin/python -m pip install -r fine_tuning/requirements.txt
```

## YOLO 데이터 준비

동일한 이름의 이미지와 라벨을 대응시킵니다.

```text
fine_tuning/data/yolo/images/train/session01_0001.jpg
fine_tuning/data/yolo/labels/train/session01_0001.txt
fine_tuning/data/yolo/images/val/session02_0001.jpg
fine_tuning/data/yolo/labels/val/session02_0001.txt
```

라벨은 사람 한 명당 `class center_x center_y width height` 한 줄입니다.
클래스는 `0`(person), 좌표는 이미지 너비·높이로 나눈 0~1 값입니다.
예: `0 0.5 0.5 0.2 0.4`. 배경 이미지는 빈 `.txt` 파일을 명시적으로 만드세요.
이 실행기는 누락 라벨을 배경으로 오인하지 않도록 라벨 파일을 요구합니다.
지원 이미지 확장자는 jpg/jpeg/png/bmp/webp입니다.

촬영 세션·날짜 단위로 train/val/test를 분리하세요. 인접 프레임을 무작위로 나누면
거의 같은 영상이 학습·검증에 동시에 들어갈 수 있습니다. 앉기·일어서기·책상 가림·
먼 거리 사례를 포함하고, test는 최종 비교까지 따로 둡니다.

## 검사 → 학습 → 검증

기본 가중치는 기존 앱이 다운로드한 `~/.cache/aix-classroom-tracking/yolo26n.pt`입니다.
다른 기존 가중치를 쓰려면 `--weights /absolute/path/model.pt`를 지정합니다.
실행기는 가중치나 데이터셋을 자동 다운로드하지 않습니다.

```sh
# 모델을 로딩하지 않고 데이터/라벨/설정/가중치 경로 검사
.venv/bin/python fine_tuning/yolo/run.py train --check

# CPU에서 연결 확인용 짧은 학습 (실제 데이터 필요)
.venv/bin/python fine_tuning/yolo/run.py train --device cpu --epochs 1

# Apple Silicon 학습 (사용 가능한 환경에서)
.venv/bin/python fine_tuning/yolo/run.py train --device mps

# NVIDIA CUDA 학습
.venv/bin/python fine_tuning/yolo/run.py train --device 0 --batch 8

# 학습한 best.pt의 탐지 성능 검증 (실제 생성된 실행 폴더로 수정)
.venv/bin/python fine_tuning/yolo/run.py val \
  --weights fine_tuning/runs/yolo/classroom-person/weights/best.pt --device cpu
```

`configs/yolo-train.yaml`에서 epochs/batch/imgsz/seed 등을 수정합니다.
여기의 상대 경로는 `fine_tuning/` 기준입니다. 데이터 YAML의 `path`는 해당 YAML 파일 기준입니다.
`--weights` 상대 경로는 명령을 실행한 현재 디렉터리 기준입니다.
기본 장치는 CPU이며 MPS/CUDA를 쓰려면 명시적으로 선택합니다.

결과는 `runs/yolo/<실행명>/`에 저장하며, 기존 폴더가 있으면 Ultralytics가 새 이름을 붙입니다.
`weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml` 등을 확인하세요.
해석된 절대 데이터 경로는 `runs/_datasets/`에 보관합니다.
중단한 학습 재개는 Ultralytics의 `YOLO('.../last.pt').train(resume=True)` API를 사용합니다.

별도 test를 준비하면 `configs/yolo-data.yaml`에 `test: images/test`를 추가하고
`val --split test --weights ...`로 최종 평가합니다. 사전학습 모델과 best.pt를 같은 split에서
검증하고 Precision/Recall/mAP와 속도를 비교하세요. 이 검증은 탐지 평가이며 IDF1/HOTA가 아닙니다.

## 추론 앱에 연결

현재 추론 앱은 지정된 출처·SHA-256의 사전학습 가중치를 로딩합니다.
학습 결과는 자동 반영되지 않습니다. 캐시의 기존 가중치를 덮어쓰지 마세요.
사용할 모델을 선정한 뒤 앱의 사용자 지정 가중치 경로 지원을 추가하거나,
검증된 새 배포 가중치의 출처·해시를 등록하는 작업이 필요합니다.
최종적으로 같은 정답 영상에서 `person_tracking/evaluate_tracking.py`의
IDF1/HOTA/ID 전환/재입장 복구율까지 비교해야 합니다.

## ReID 후속 작업

[AIM 준비](aim/README.md), [BPBreID 준비](bpbreid/README.md)를 참고하세요.
얼굴 모델은 현재 우선 학습 대상에 포함하지 않았습니다.

공식 문서: [학습 API](https://docs.ultralytics.com/modes/train/),
[탐지 데이터 형식](https://docs.ultralytics.com/datasets/detect/).
