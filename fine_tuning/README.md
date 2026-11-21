# 강의실 상반신 탐지 파인튜닝

COCO 사전학습 YOLO26n으로 classroom-dodzk v1의 상반신 person 한 클래스를 학습합니다.
원본 1,650장 중 1,274장을 정제해 학습 764장, 검증 275장, 테스트 235장에 사용했습니다.
[상반신 주석 기준과 데이터 출처](ANNOTATION_POLICY.md)

## 테스트 결과

같은 테스트 이미지 235장을 사용했습니다.

| 모델 | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| COCO 사전학습 YOLO26n | 85.48% | 53.81% | 71.15% | 50.89% |
| classroom-dodzk 파인튜닝 | 98.49% | 98.81% | 99.31% | 84.58% |

mAP50–95는 33.69%p 올랐습니다. 원본 분할에서 같은 강의실이나 인접 프레임이 여러 분할에
포함될 수 있습니다. 영상 ID 추적과 출석 정확도는 별도 평가가 필요합니다.

## 데이터 준비와 실행

원본 Roboflow 내보내기는 `fine_tuning/data/roboflow-classroom-dodzk/raw/`에 둡니다.
정제본은 같은 경로의 `person/` 폴더에 생성합니다. 기본 시작 가중치는 앱이 사용한
`~/.cache/aix-classroom-tracking/yolo26n.pt`입니다.

프로젝트 루트에서 실행합니다. 아래의 `--device 0`은 CUDA 장치이며, 환경에 따라
`--device mps` 또는 `--device cpu`를 지정할 수 있습니다.

```sh
.venv/bin/python -m pip install -r fine_tuning/requirements.txt
.venv/bin/python fine_tuning/yolo/prepare_classroom_dodzk.py
.venv/bin/python fine_tuning/yolo/run.py train --check
.venv/bin/python fine_tuning/yolo/run.py train --device 0

# 동일 테스트 분할에서 학습 전후 모델 평가
.venv/bin/python fine_tuning/yolo/run.py val --split test --device 0
.venv/bin/python fine_tuning/yolo/run.py val --split test --device 0 --weights fine_tuning/runs/yolo/classroom-dodzk-person/weights/best.pt
```

학습 설정은 `configs/classroom-dodzk-train.yaml`에 있습니다. 여러 번 학습했다면
실제 실행 폴더의 가중치 경로를 지정합니다. 학습한 가중치를 앱에 적용하려면 별도로 연결해야 합니다.

## 입력 크기 640과 768 비교

모델 A는 기본 640 설정, 모델 B는 아래의 768 설정을 사용합니다. 두 설정은 같은
데이터 분할과 사전학습 YOLO26n을 사용합니다. 모델을 고를 때는 각 실행의
`results.csv`에서 검증 mAP50–95를 비교하고, 선택 후 테스트 235장에서 성능을 확인합니다.

```sh
.venv/bin/python fine_tuning/yolo/run.py train --config configs/classroom-dodzk-768-train.yaml --check
.venv/bin/python fine_tuning/yolo/run.py train --config configs/classroom-dodzk-768-train.yaml --device 0

# 모델 B가 검증에서 선정됐을 때의 최종 테스트
.venv/bin/python fine_tuning/yolo/run.py val --config configs/classroom-dodzk-768-train.yaml --device 0 --split test --weights fine_tuning/runs/yolo/classroom-dodzk-person-768/weights/best.pt
```
