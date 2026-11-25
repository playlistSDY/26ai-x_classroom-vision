# 최종 교실 상반신 탐지 모델

**COCO 기본학습 YOLO26n을 교실 및 상반신 데이터셋 2개로 통합 파인튜닝했습니다.**
완료 실험은 `upper-body-unified-v2`이며, 최종 가중치는
[배포본의 models 폴더](../deploy/models/upper-body-unified-v2-best.pt)에 동봉했습니다.
[시연 노트북](../deploy/START.ipynb)은 이 모델을 기본으로 사용합니다.

## 왜 파인튜닝했는가

교실에서는 책상·노트북·앞사람에 가려져 전신보다 머리와 상반신만 보이는 경우가 많습니다.
이 조건에 맞춰 **한 사람의 머리와 보이는 어깨·몸통을 감싸는 person 박스 하나**로
두 데이터셋의 주석 기준을 통일했습니다. 출발점은 COCO 사전학습 YOLO26n입니다.
AIM-LTCC·BPBreID·얼굴 모델은 사전학습 가중치를 사용합니다.

## 데이터 출처와 구성

| 데이터셋 | 원본 | 정제 후 | 정제 내용 |
| --- | ---: | ---: | --- |
| [classroom-dodzk v1](https://universe.roboflow.com/reddy-projects-zulke/classroom-dodzk/dataset/1) | 1,650장 | 1,274장 | 원본 클래스 8·9의 상반신 박스를 person으로 통합 |
| [ClassRoom v1](https://universe.roboflow.com/chokoppang/classroom-o4dnp) · [사용한 포크](https://app.roboflow.com/buenos-dias/classroom-o4dnp-k83mj/1) | 450장 | 253장 | 여러 사람·책상을 감싼 박스, 누락·중복을 검토하고 일부 재표시 |
| **합계** | **2,100장** | **1,527장** | **19,012개 person 박스** |

두 출처의 제공 메타데이터에 표시된 라이선스는 CC BY 4.0입니다.

| 분할 | 교실 상반신 | 교실 정제 | 합계 | person 박스 |
| --- | ---: | ---: | ---: | ---: |
| 학습 | 764 | 176 | 940 | 11,635 |
| 검증 | 275 | 51 | 326 | 4,819 |
| 테스트 | 235 | 26 | 261 | 2,558 |
| 전체 | 1,274 | 253 | 1,527 | 19,012 |

원본 분할을 유지했으며 이미지 SHA-256의 정확한 중복은 없습니다.
교실 정제 데이터에는 사람이 없는 배경 16장도 포함했습니다.
상세 주석 기준과 검수 범위는 [ANNOTATION_POLICY.md](ANNOTATION_POLICY.md)에 있습니다.

## 학습 조건

| 항목 | 설정 |
| --- | --- |
| 출발 모델 | COCO 사전학습 YOLO26n · `yolo26n.pt` |
| 최종 클래스 | `0: person` |
| 장치 | NVIDIA RTX A2000 6GB |
| 환경 | PyTorch 2.11.0+cu128 · Ultralytics 8.4.202 |
| 입력 / 배치 / workers | 640 / 4 / 2 |
| 최대 epochs / patience | 50 / 10 |
| 완료 / 최적 검증 epoch | 46 / 36 |
| 최종 모델 선택 | 검증 mAP50–95 |
| seed / deterministic | 42 / true |

학습 설정은 [YAML](configs/upper-body-unified-v2-train.yaml), 실제 전체 인자는
[args.yaml](reports/unified-v2-logs/args.yaml)에 보관했습니다.

## COCO 기본학습과 최종 파인튜닝 비교

출처별 동일 정제 테스트에서 입력 640·배치 4·RTX A2000·`person` 클래스로 평가했습니다.

| 평가 데이터 | 모델 | Precision | Recall | mAP50 | mAP50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 교실 상반신 · 235장 | COCO 기본학습 | 85.48% | 53.81% | 71.15% | 50.89% |
| 교실 상반신 · 235장 | 교실 및 상반신 데이터셋 2개 통합 파인튜닝 | **98.14%** | **98.86%** | **99.27%** | **84.13%** |
| 교실 정제 · 26장 | COCO 기본학습 | 35.44% | 62.32% | 46.07% | 12.66% |
| 교실 정제 · 26장 | 교실 및 상반신 데이터셋 2개 통합 파인튜닝 | **100.00%** | **98.97%** | **99.50%** | **72.84%** |

mAP50–95는 각각 **33.24%p·60.18%p**, Recall은 **45.05%p·36.66%p** 향상됐습니다.
향상 폭은 원본 소수값의 차이로 계산했습니다.

같은 장소·인접 프레임이 분할에 포함되고 교실 정제 테스트는 26장입니다.
첫 데이터는 기존 정제본을 계승해 이번에는 12장을 표본 재검수했으며,
두 번째 데이터는 450장을 시각 선별한 뒤 일부를 직접·모델 보조로 재표시했습니다.
평가 범위는 정지 이미지의 상반신 탐지입니다. 영상의 ID 유지·인원 집계에는 별도 정답 평가가 필요합니다.

## 최종 가중치와 실행

- 동봉 파일: [`../deploy/models/upper-body-unified-v2-best.pt`](../deploy/models/upper-body-unified-v2-best.pt)
- SHA-256: `857c7d671b47292bf81bcae961d230f1b7913dc5894abdef8e626ddabff0edf2`
- 시연 실행: [`../deploy/START.ipynb`](../deploy/START.ipynb) 전체 실행
- 설치·장치 선택·카메라·영상 사용: [배포본 README](../deploy/README.md)

이 가중치는 `deploy` 실행본의 기본 탐지 모델입니다.
원본 `person_tracking` 개발 앱의 캐시 가중치는 별도로 유지합니다.

## 재현 자료

| 자료 | 위치 |
| --- | --- |
| 데이터 주석·정제 기준 | [ANNOTATION_POLICY.md](ANNOTATION_POLICY.md) |
| 데이터 구성 | [데이터 YAML](configs/upper-body-unified-v2-data.yaml) |
| 학습 설정 | [학습 YAML](configs/upper-body-unified-v2-train.yaml) |
| 실제 46에포크 학습·검증 | [CSV](reports/unified-v2-training.csv) |
| 테스트 원본 수치·분할·해시 | [실험 요약 JSON](reports/unified-v2-20261119.json) |
| 실제 학습 인자·콘솔·출처 확인 | [학습 로그 안내](reports/unified-v2-logs/README.md) |
| 기존 중간 실험 포함 기록 | [UNIFIED_TRAINING_RESULTS.md](UNIFIED_TRAINING_RESULTS.md) |
| 동일 영상의 과거 모델 비교 | [영상 비교 기록](reports/classroom-video-20261122/summary.json) |

원본 데이터와 이미지별 검수 원장은 별도 보관합니다. 동일한 정제본을 재생성하려면
원본 데이터와 `data/upper-body-unified-v1-review/decisions.json`이 필요합니다.
정제본 manifest와 검수 원장의 SHA-256은 실험 요약 JSON에 기록했습니다.

### 학습 재실행

프로젝트 루트에서 학습 환경을 준비합니다. GPU용 PyTorch는 장치에 맞게 설치합니다.

```sh
python3.12 -m venv .venv
```

macOS·Linux는 `source .venv/bin/activate`, Windows PowerShell은
`.venv\Scripts\Activate.ps1`로 활성화합니다.

```sh
python -m pip install -r fine_tuning/requirements.txt
```

위에서 만든 가상환경을 활성화한 뒤 다음 순서로 실행합니다.
원본·검수 원장과 COCO 기본 가중치 `yolo26n.pt`를 먼저 준비합니다.

```sh
# 첫 데이터 정제본이 이미 있으면 생략
python fine_tuning/yolo/prepare_classroom_dodzk.py

# 두 데이터의 통합 정제본 구성
python fine_tuning/yolo/prepare_unified_classroom.py

# 데이터와 설정 사전 검사
python fine_tuning/yolo/run.py train \
  --config configs/upper-body-unified-v2-train.yaml \
  --weights /path/to/yolo26n.pt --device 0 --check

# 최종 통합 모델 학습
python fine_tuning/yolo/run.py train \
  --config configs/upper-body-unified-v2-train.yaml \
  --weights /path/to/yolo26n.pt --device 0
```

`/path/to/yolo26n.pt`는 실제 COCO 기본 가중치 경로로 바꿉니다.
학습 출력은 `fine_tuning/runs/yolo/upper-body-unified-v2/`이며 기존 출력은 보호합니다.
새 실험은 학습 YAML의 `name`을 바꿔 실행합니다.

`train_unified_classroom.py`는 중간 모델까지 포함한 당시 세 모델 비교용 실행기입니다.
현재 최종 모델만 학습하려면 위의 `run.py train`을 사용합니다.
학습 준비 검사는 `python -m unittest discover -s fine_tuning/tests -v`로 실행합니다.
