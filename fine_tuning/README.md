# 강의실 사람 탐지 파인튜닝

원래 앱에서 사용한 **COCO 사전학습 YOLO26n**을 출발점으로, 두 강의실 데이터셋의
주석을 상반신 `0: person` 기준으로 정리해 함께 학습했습니다. 완료된 최종 실험은
`upper-body-unified-v2`입니다. 사람 탐지 학습이며 AIM-LTCC·BPBreID·얼굴 모델은
이번 실험에서 추가 학습하지 않았습니다.

## 데이터 출처와 정제

| 데이터 | 원본 | 사용 방식 | 통합본에 채택 |
|---|---:|---|---:|
| [reddy projects / classroom-dodzk v1](https://universe.roboflow.com/reddy-projects-zulke/classroom-dodzk/dataset/1) | 1,650장 | 클래스 이름 8·9의 상반신만 person으로 통합. 머리 중심에 대응하는 상반신이 없는 이미지 376장 제외 | 1,274장 |
| [buenos-dias / ClassRoom v1](https://app.roboflow.com/buenos-dias/classroom-o4dnp-k83mj/1) | 450장 | Student·with-student 주석을 시각 선별. 책상·여러 사람 포함 박스와 누락·중복을 확인해 일부 재표시, 불확실한 이미지 197장 제외 | 253장 |

두 데이터의 제공 메타데이터에 표시된 라이선스는 CC BY 4.0입니다. 첫 데이터의
숫자 클래스는 사람별 ID가 아니며, 8·9의 역할은 주석 샘플의 박스 범위를 확인한
해석입니다. 두 번째 데이터의 두 클래스를 그대로 person으로 일괄 변환하지 않았습니다.

기준은 **한 사람의 머리와 보이는 어깨·몸통을 감싸는 상반신 박스 하나**입니다.
얼굴 박스, 손·팔만 보이는 박스, 책상 전체, 여러 사람을 함께 감싸는 박스를 같은
person 정답으로 섞지 않습니다. 얼굴은 기존 YuNet/SFace 추론에서 ID 보조 정보로
사용합니다. 상세 기준과 검수 범위는 [주석 기준](ANNOTATION_POLICY.md)을 참고하세요.

| 분할 | 첫 데이터 | 두 번째 데이터 | 합계 | person 박스 |
|---|---:|---:|---:|---:|
| train | 764 | 176 | 940 | 11,635 |
| valid | 275 | 51 | 326 | 4,819 |
| test | 235 | 26 | 261 | 2,558 |
| 전체 | 1,274 | 253 | 1,527 | 19,012 |

원본 분할을 유지했습니다. 두 번째 데이터에는 사람이 없는 것을 확인한 배경 이미지
16장(train 11·valid 3·test 2)이 포함됩니다. 이미지 SHA-256의 정확한 중복은 없습니다.

**검수 범위:** 첫 데이터는 기존 정제본을 계승했고 이번 시각 재검수는 표본 12장입니다.
두 번째 데이터는 450장 시각 선별 후 일부를 직접 또는 모델 보조로 재표시했습니다.
두 데이터 전체를 새로 재주석한 독립 정답셋은 아닙니다. 같은 장소·인접 프레임이
분할 간에 나타나므로 실제 설치 장소의 일반화 성능과 구분해야 합니다.

## 학습 설정과 완료 결과

두 모델을 따로 학습한 뒤 가중치를 합친 것이 아니라, **정제한 두 데이터셋을 합쳐
원래 YOLO26n에서 하나의 모델을 학습**했습니다. 이전 person 파인튜닝 모델에
이어 학습한 실험도 아닙니다. 설정은 [학습 YAML](configs/upper-body-unified-v2-train.yaml)에 있습니다.

| 항목 | 실제 조건 |
|---|---|
| 출발 가중치 | 앱이 원래 사용하던 `yolo26n.pt` |
| 입력 크기 / 배치 | 640 / 4 |
| 최대 epochs / patience | 50 / 10 |
| 실제 완료 / 최적 검증 epoch | 46 epoch에서 조기 종료 / 36 epoch |
| workers / seed | 2 / 42 |
| 장치 | NVIDIA RTX A2000, VRAM 약 6 GB |
| 환경 | Python 3.12.15, PyTorch 2.11.0+cu128, Ultralytics 8.4.202 |

## 원래 사용하던 모델과 비교

아래 **기존 모델은 사전학습 YOLO26n**입니다. 각 행의 기존·통합 모델은 **같은 정제
테스트 이미지·정답, 입력 640, batch 4, CUDA, 예측 클래스 0(person)**으로 평가했습니다.
수치는 비율의 백분율이며 mAP50–95는 IoU 0.50~0.95 평균입니다.

| 테스트 | 모델 | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|
| classroom-dodzk 235장 | 기존 YOLO26n | 85.48% | 53.81% | 71.15% | 50.89% |
| classroom-dodzk 235장 | **통합 파인튜닝** | **98.14%** | **98.86%** | **99.27%** | **84.13%** |
| ClassRoom 정제본 26장 | 기존 YOLO26n | 35.44% | 62.32% | 46.07% | 12.66% |
| ClassRoom 정제본 26장 | **통합 파인튜닝** | **100.00%** | **98.97%** | **99.50%** | **72.84%** |

- 첫 데이터: mAP50–95 **+33.24%p**, Recall **+45.05%p**.
- 새 정제 데이터: mAP50–95 **+60.18%p**, Recall **+36.66%p**.
- Precision 100%는 선별된 작은 테스트셋에서의 결과이며 오류 없는 시스템을 뜻하지 않습니다.
- 기존 모델은 일반 person 박스를, 정제 정답은 상반신을 대상으로 하므로 박스 범위 차이가
  특히 mAP50–95에 영향을 줍니다. 결과는 이 상반신 탐지 과제에 대한 적합도를 나타냅니다.

중간에 첫 데이터만 학습한 person 모델과도 비교했습니다. 첫 데이터의 mAP50–95는
84.58% → 84.13%로 0.45%p 감소했고, 새 정제 데이터는 10.37% → 72.84%로 62.47%p
증가했습니다. 통합 모델이 모든 조건에서 더 좋아졌다고 해석하지 않습니다.
전체 세 모델 비교는 [상세 결과](UNIFIED_TRAINING_RESULTS.md), 수치·환경·해시는
[JSON 요약](reports/unified-v2-20261119.json)에 보관합니다.

이 결과는 **이미지의 상반신 탐지** 평가입니다. 출석 정확도·IDF1·HOTA·재입장 복구율,
실시간 FPS 개선은 이번 학습으로 검증하지 않았습니다. 과거 ClassRoom 원본의 두 클래스
모델 평가도 정답 기준과 테스트 이미지가 달라 직접 비교할 수 없습니다.

## 파일 구조와 데이터 준비

```text
fine_tuning/
  ANNOTATION_POLICY.md        # 주석 기준과 검수 범위
  UNIFIED_TRAINING_RESULTS.md # 세 모델 비교
  configs/                   # 데이터 경로와 학습 설정
  reports/                   # 공개 가능한 집계 지표·설정·해시
  yolo/                      # 준비·학습·평가 코드
  tests/                     # 데이터 준비 보호 장치 테스트
  data/                      # 원본·정제본·검수 원장 (Git 제외)
  checkpoints/               # 가중치 (Git 제외)
  runs/                      # 학습 로그·평가·실행 설정 (Git 제외)
```

저장소에는 데이터셋, 식별 가능한 이미지·영상, 이미지별 검수 원장, 모델 가중치,
API 키를 포함하지 않습니다. 내려받은 원본과 검수 자료는 별도로 준비해야 합니다.
검수 원장은 `data/upper-body-unified-v1-review/decisions.json`에 두며, 실행기는 원본
450장에 대해 누락·중복 없는 명시적 채택/제외 결정과 채택 박스를 요구합니다.
**원본 ZIP만으로는 이번 사람이 확인한 정제본을 완전히 재생성할 수 없습니다.**
동일 실험 재현에는 보관 중인 검수 원장이 필요합니다. 요약 JSON의 manifest·원장
해시로 같은 정제본인지 확인하세요. 검수 원장이 없으면 주석 기준에 따라 새 원장을
만들고 별도 실험으로 평가해야 합니다.

학습 전용 환경은 기존 추적 환경의 Python을 재사용하거나 다음처럼 준비합니다.
GPU용 PyTorch는 장치에 맞는 버전을 별도로 설치합니다.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r fine_tuning/requirements.txt
```

로컬 자료 위치:

```text
fine_tuning/data/roboflow-classroom-dodzk/raw/
fine_tuning/data/roboflow-classroom-dodzk/person/
fine_tuning/data/shared-classroom-v1/raw/
fine_tuning/data/upper-body-unified-v1-review/decisions.json
fine_tuning/data/upper-body-unified-v2/
~/.cache/aix-classroom-tracking/yolo26n.pt
```

실행기는 데이터와 가중치를 자동 다운로드하지 않습니다.

## 준비 → 검사 → 학습 → 출처별 평가

아래는 프로젝트 루트에서 실행합니다. 기존 정제본·완료 실험을 보존하므로 준비나
실험 출력이 이미 있으면 새 출력 경로·이름을 사용하세요.

```sh
# 첫 데이터의 기존 상반신 정제 (이미 있으면 실행 생략)
.venv/bin/python fine_tuning/yolo/prepare_classroom_dodzk.py

# 두 원본과 별도 보관된 검수 원장으로 통합본 생성
.venv/bin/python fine_tuning/yolo/prepare_unified_classroom.py

# 모델을 로딩하거나 학습을 시작하지 않는 데이터 사전 검사
.venv/bin/python fine_tuning/yolo/run.py train \
  --config configs/upper-body-unified-v2-train.yaml --device 0 --check

# 학습과 세 모델의 출처별 테스트 비교
.venv/bin/python -u fine_tuning/yolo/train_unified_classroom.py
```

비교 실행기는 첫 person 모델이 아래 위치에 있을 것을 요구합니다.
그 모델도 없다면 첫 정제본으로 먼저 생성합니다.

```sh
.venv/bin/python fine_tuning/yolo/run.py train \
  --config configs/classroom-dodzk-train.yaml --device 0
```

- 첫 person 모델: `fine_tuning/runs/yolo/classroom-dodzk-person/weights/best.pt`
- 통합 원본 결과: `fine_tuning/runs/yolo/upper-body-unified-v2/weights/best.pt`
- 통합 복사본: `fine_tuning/checkpoints/upper-body-unified-v2-best.pt`
- 상태·출처별 지표: `fine_tuning/runs/upper-body-unified-v2-20261119/status.json`
- 실제 학습 인자: `fine_tuning/runs/yolo/upper-body-unified-v2/args.yaml`

원래 분리 실험인 `yolo/train_shared_classroom.py`는 원본의 Student·with-student
두 클래스를 유지한 모델을 만듭니다. 현재 통합 person 모델을 재현하는 명령이 아닙니다.
해당 원본 실험에는 `fine_tuning/data/shared-classroom-v1/ClassRoom.v1i.yolo26.zip`도 필요합니다.

## 검증과 앱 적용 상태

```sh
.venv/bin/python -m unittest discover -s fine_tuning/tests -v
```

18개 테스트를 통과했습니다. 잘못된 주석·누락 라벨·중복 이미지·미검수 후보의 반영을
거부하고, 제외 이미지를 배경으로 바꾸지 않으며, 기존 출력과 원본을 보존하는지 검사합니다.

학습한 모델은 현재 앱의 기본 가중치를 자동 교체하지 않습니다. 기존 캐시를 덮어쓰지
마세요. 이 브랜치는 학습·정제·평가와 기록을 추가하며, 실제 앱 연결 및 정답 영상의
ID 유지 검증은 별도 작업입니다.

통합 가중치 SHA-256:
`857c7d671b47292bf81bcae961d230f1b7913dc5894abdef8e626ddabff0edf2`
