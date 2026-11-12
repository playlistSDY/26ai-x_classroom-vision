# 실시간 사람 ID 추적

카메라 영상을 처리해 실행 중인 세션에서 사람별 ID를 유지합니다.
BPBreID 구성요소에는 조건이 있는 [Hippocratic License 3.0](vendor/LICENSE.bpbreid)이
적용됩니다. 출처와 수정 사항은 [NOTICE](vendor/NOTICE.md)를 확인하세요.

프로젝트 루트의 터미널에서 실행합니다.

```sh
.venv/bin/python -m pip install -r person_tracking/requirements.txt
.venv/bin/python person_tracking/person_tracking.py --camera 1
```

이 Mac에서 확인한 카메라는 1번이 iPhone, 0번이 내장 카메라입니다.
연결 상태에 따라 번호가 바뀔 수 있습니다. 이름 입력란 밖에서 Q 또는 Esc로 종료합니다.
CUDA, Apple Silicon MPS, CPU를 자동 선택하며 `--device mps` 등으로 지정할 수 있습니다.
최초 실행에는 사람 탐지·외형·부분 신체·얼굴 모델을 다운로드합니다.

## AIM-LTCC 비활성화 비교

`--no-aim`을 지정하면 AIM 가중치를 로딩·다운로드하지 않고 전신 외형 특징 추론도
건너뜁니다. YOLO, BoT-SORT, BPBreID와 얼굴 비교는 유지합니다. 옵션을 생략하면
기존 5개 모델 구성이며, 판정 임계값은 두 구성에서 동일하게 유지합니다.

```sh
.venv/bin/python person_tracking/person_tracking.py --camera 0 --no-aim

# 같은 영상·정답·장치로 비교
.venv/bin/python person_tracking/evaluate_tracking.py \
  --video data/classroom.avi --ground-truth data/gt.txt --output outputs/aim-on.json
.venv/bin/python person_tracking/evaluate_tracking.py \
  --video data/classroom.avi --ground-truth data/gt.txt --no-aim --output outputs/aim-off.json
```

평가 JSON의 `protocol.aim_enabled`와 MLflow의 `aim_enabled`에 활성화 여부를 기록합니다.
MLflow 모델 목록·가중치 업로드에서도 비활성화한 AIM은 제외합니다.
속도와 함께 IDF1, ID Switches, HOTA, 재입장 ID 복구율을 비교하세요.

## MLflow 기록 (선택)

```sh
.venv/bin/python -m pip install -r person_tracking/requirements-mlflow.txt
.venv/bin/python person_tracking/person_tracking.py --camera 0 --mlflow --run-name baseline
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

마지막 명령은 다른 터미널에서 실행하고 <http://127.0.0.1:5000>에 접속합니다.
실행 설정, 집계 지표, 모델 출처와 해시를 기록하며 `--log-models`를 추가하면
가중치도 보관합니다. 얼굴 특징·영상·이름은 전송하지 않습니다.
설치, 지표 정의, 모델 다운로드와 공용 서버 연결은
[프로젝트 README의 MLflow 안내](../README.md#mlflow로-실행과-모델-관리)를 참고하세요.

## 정답 기반 영상 평가

Precision / Recall, IDF1, ID Switches, HOTA, Re-ID Accuracy를 계산하려면 **사람별 정답 ID와
박스를 모든 프레임에 라벨링한 영상**이 필요합니다. 정답 없는 카메라 실행에서는 이 값들을
추정하거나 만들어 내지 않습니다. Actual FPS는 카메라 실행에도 자동 기록됩니다.

### 실행

프로젝트 루트에서 평가 의존성을 설치하고 실제 영상·정답 경로를 지정합니다.

```sh
.venv/bin/python -m pip install -r person_tracking/requirements-evaluation.txt
.venv/bin/python person_tracking/evaluate_tracking.py \
  --video data/classroom.avi --ground-truth data/gt.txt \
  --device auto --size 768 \
  --mlflow --tracking-uri http://127.0.0.1:5000 \
  --experiment-name classroom-vision --run-name classroom-evaluation \
  --output outputs/classroom-evaluation.json
```

대시보드는 별도 실행해야 합니다. `--tracking-uri`를 생략하면 기존 환경변수/로컬 SQLite
설정을 사용합니다. `--mlflow`와 관련 옵션을 모두 생략하면 MLflow 기록 없이 콘솔과 JSON으로
평가 결과만 확인할 수 있습니다. 모델은 동일하게 다운로드·해시 검증하며 GUI나 카메라는 필요 없습니다.
`--reid-interval`, `--reid-batch`, `--log-models`도 기존 앱과 같은 의미입니다.

### 정답 파일 형식

헤더 없는 MOT 스타일 9열 CSV를 사용합니다.

```text
frame,id,x,y,width,height,mark,class,visibility
```

파일에는 위 헤더를 넣지 않습니다. 예를 들어 한 사람이 1·2프레임에 보이고,
3·4프레임에는 완전히 가려졌다가 5프레임에 같은 사람으로 돌아오면 다음과 같습니다.

```text
1,1,100,120,80,180,1,1,1
2,1,102,120,80,180,1,1,0.8
5,1,110,120,80,180,1,1,1
```

- `frame`은 영상 첫 프레임을 1로 세는 정수입니다. 프레임을 건너뛰지 않고 전체 영상을 평가합니다.
- `id`는 정답 인물의 양의 정수 ID입니다. 이탈·가림 후에도 같은 사람은 같은 ID를 사용합니다. 앱의 P001과 번호가 같을 필요는 없습니다.
- 박스는 영상 원본 해상도의 픽셀 좌표이며 왼쪽 위 `(x, y)`와 양수 크기 `(width, height)`입니다. 좌표 원점은 `(0, 0)`입니다.
- 이 평가기는 사람 전용으로 정리된 `mark=1`, `class=1` 라벨만 받습니다. ignore/distractor 라벨은 오류로 처리하며 MOTChallenge의 공식 전처리 전체를 구현한 것은 아닙니다. 그런 데이터셋은 공식 평가 파이프라인을 사용하세요.
- `visibility`는 0~1 값입니다. 가시성으로 라벨을 제외하지 않고 파일에 있는 모든 사람 박스를 평가합니다.
- **행이 없는 프레임은 사람이 없는 정답 프레임**입니다. 일부 프레임만 라벨링한 파일을 넣으면 누락 프레임의 탐지가 오탐으로 계산됩니다. 영상과 라벨을 같은 구간으로 잘라 정렬하세요.
- 예제 라벨은 형식 설명용입니다. 저장소에는 실제 강의실 정답 데이터가 포함되어 있지 않습니다.

### 지표 해석

MLflow의 `precision`, `recall`, `idf1`, `hota`, `reid_accuracy`는 0~1 값입니다.
Precision / Recall은 중복 제거 후 YOLO 사람 박스에 대해 IoU 기준 일대일 매칭으로 계산합니다.
IDF1 / ID Switches / HOTA는 실제 확정 세션 ID를 사용해 [TrackEval](https://github.com/JonathonLuiten/TrackEval)의
Identity / CLEAR / HOTA 구현으로 계산합니다. 설치 패키지는 NumPy 호환 수정이 적용된
[PyPI TrackEval 1.3.0 배포판](https://pypi.org/project/trackeval/1.3.0/)입니다.
`Checking` 박스는 탐지 평가에 포함하지만 ID 평가에는 확정 ID 예측이 없으므로 누락으로 반영합니다.

`--iou-threshold 0.5`로 탐지 매칭, IDF1, ID Switches, Re-ID 공간 매칭의 기준을 지정합니다.
HOTA는 이 옵션과 별개로 표준 IoU 0.05~0.95의 19개 기준을 평균합니다.

**이 프로젝트의 Re-ID Accuracy는 검색 벤치마크의 Rank-1 / mAP가 아니라 재입장 ID 복구율입니다.**
정답 인물이 `--reid-gap-frames 1` 이상의 프레임 동안 정답에서 사라졌다가 돌아오면 복귀 이벤트로 셉니다.
이전까지 IoU 매칭된 확정 세션 ID가 있는 이벤트만 분모(`reid_events`)에 포함합니다.
첫 복귀 프레임에서 그 ID를 되찾으면 `reid_correct`가 증가하며,
새 ID, `Checking`, 미검출이면 실패입니다. 나중에 복구해도 첫 복귀 실패를 소급 수정하지 않습니다.
이전 ID가 없어 평가에서 제외된 복귀는 `reid_ineligible_returns`로 따로 기록합니다.
평가 가능한 복귀가 없으면 JSON은 `reid_accuracy: null`, MLflow에는 해당 키를 기록하지 않습니다.
정답에 계속 보이는데 모델만 놓친 구간은 이 지표의 복귀 이벤트가 아닙니다.

### 실제 FPS와 결과 보관

`actual_fps = pipeline_frames / pipeline_seconds`이며 `1000 / mean_process_ms`와 다릅니다.
모델 로딩을 끝낸 뒤 입력 수집/읽기를 시작한 시점부터 마지막 프레임 처리 완료까지 측정합니다.
카메라는 대기, Qt 화면 갱신, 주기적 MLflow 통신과 프레임 사이 지연을 포함합니다.
카메라 캡처는 최신 프레임만 처리하므로 카메라 원본 FPS가 아니라 **앱의 처리 완료 FPS**입니다.
세션 초기화로 FPS 통계를 초기화하지 않으며 마지막 프레임 이후 종료 대기는 제외합니다.
오프라인 영상은 디코딩, 탐지·재식별·추적, 평가 입력 누적, 주기적 로깅을 포함하지만 GUI는 없습니다.
최종 HOTA 계산, 파일 해시 계산, 결과 JSON/최종 아티팩트 저장은 FPS 측정에서 제외합니다.

결과 JSON에는 TP/FP/FN, 재입장 이벤트 수, HOTA의 IoU별 점수, 평가 기준,
영상·정답 SHA-256과 TrackEval 버전이 포함됩니다. MLflow 사용 시 `evaluation/summary.json`으로도
보관합니다. 원본 영상·정답 박스·사람별 ID 목록은 업로드하지 않습니다.
전체 시퀀스의 박스·ID를 메모리에 보관해 평가하므로 긴 영상은 평가용 구간으로 나눠 사용하세요.

## 카메라 창 조작

- **현재 인원**은 화면에 추적 박스가 표시된 인원입니다. ID를 확인 중인 박스도 포함합니다.
- **누적 인원**은 이번 세션에 등록된 고유 ID 수입니다. 기존 ID로 재입장하면 추가로 세지 않습니다.
- 영상 속 박스나 오른쪽 ID 목록을 선택하고 이름을 입력한 뒤 **이름 저장**을 누릅니다.
  이름은 ID와 함께 표시합니다. 이름을 비운 채 저장하면 ID만 표시합니다.
- **라벨 및 인원 기록 초기화**를 확인하면 등록 ID, 이름, 인원 집계가 초기화되고 P001부터 다시 등록합니다.
- 영상 속 박스를 누르면 **ID 진단** 탭이 열립니다. `Checking` 박스도 선택할 수 있습니다.
  등록된 사람을 목록에서 고른 뒤 진단 탭을 열면 화면 밖 인물의 마지막 관측도 확인할 수 있습니다.
  고정된 비교표에서 후보별 얼굴·외형·신체 부위 유사도와 판정값을 확인합니다.
  **최근 비교**와 **번호 결정**을 선택하면 같은 표에서 비교 시점만 바뀝니다.
  최근 비교의 얼굴 열은 마지막 얼굴 측정값을 표시합니다. 새 얼굴 측정에서 점수를 얻지 못하면
  얼굴 열을 비우고 표 아래에 미검출, 흐림, 사람 박스 연결 불확실, 등록 얼굴 없음 등의 상태를 표시합니다.

## ID를 연결하는 방법

- YOLO26n으로 사람을 찾고, BoT-SORT로 프레임 사이의 움직임을 연결합니다.
- YuNet으로 얼굴을 찾고, SFace의 사전 학습된 얼굴 특징을 ID별로 누적합니다.
  선명한 얼굴이 있으면 얼굴을 먼저 비교하며, 얼굴이 기존 사람과 다르면
  옷이 비슷하다는 이유로 같은 ID를 주지 않습니다.
- 얼굴이 가려지거나 뒤돌아 있으면 외형·신체 부위·이동 경로로 이어갑니다.
  얼굴로 ID가 확인되면 바뀐 옷의 모습도 여러 프레임에서 확인해 저장합니다.
- 함께 보이는 별도 사람은 옷이 비슷해도 서로 다른 ID로 등록합니다.
  추적 중인 ID를 외형 점수가 조금 높다는 이유로 다른 사람에게 넘기지 않습니다.
- AIM-LTCC의 외형 특징과 여러 프레임에서 확인한 각도·가림 상태를 사람별로 저장합니다.
- BPBreID는 머리, 몸통, 좌우 팔·다리·발의 특징과 가시성을 추정합니다.
  상체만 보이는 경우에는 과거 전신 관측의 공통 부위와 비교합니다.
- 다른 사람의 박스와 겹친 픽셀은 부분 특징의 풀링에서 제외합니다.
  책상과 구조물은 BPBreID가 학습한 배경·신체 주의 맵으로 처리합니다.
- 부분 특징으로 재입장을 연결하려면 두 개 이상의 신체 그룹과 여러 번의
  관측이 필요합니다. 좌우 다리만 보이는 것은 한 그룹으로 셉니다.
- 신체 부위 단서가 부족해도 이미 저장한 외형과 비교할 수 있습니다.
  단서가 부족한 상태에서 새 번호를 발급하는 것은 보류합니다.
- 확신이 부족하면 `Checking`을 표시합니다. 한 화면에서 두 사람에게 같은
  ID를 부여하지 않습니다.

처음 몇 초 동안 선명하게 여러 방향을 보여주면 저장되는 특징이 늘어납니다.
ID와 특징은 메모리에 보관하며 프로그램을 다시 시작하면 초기화됩니다.
얼굴 모듈은 OpenCV CPU에서 실행하고 몸 특징 추론은 선택한 MPS/CUDA/CPU를 사용합니다.
얼굴 모델의 출처와 라이선스는 [얼굴 모델 안내](vendor/NOTICE.face-models.md)에 있습니다.
완전한 가림, 아주 작은 신체 조각, 공통으로 보이는 부위가 없는 재입장,
비슷한 외형의 사람은 자동 연결이 어려울 수 있습니다. 신체 가시성도
모델의 추정값이며 실제로 가려진 모든 픽셀을 정확하게 판별하는 것은 아닙니다.

## 검증

```sh
.venv/bin/python -m unittest discover -s person_tracking/tests -v
```

테스트는 ID 유지, 재입장, 신체 부위 정렬, 불확실한 관측, 중복 박스,
다른 사람의 특징 혼입 방지를 확인합니다. 합성 가림과 공개 보행 영상으로
구성한 검증은 실제 강의실의 앉기·일어서기·가림 성능을 대신하지 않습니다.

부분 특징의 초기 기준은 작은 공개 영상 샘플에서 확인한 보수적인 값입니다.
실제 강의실 성능이나 모든 각도·복장·가림에 대한 인식률을 보장하지 않습니다.

모델 출처: [AIM-CCReID](https://github.com/BoomShakaY/AIM-CCReID),
[BPBreID](https://github.com/VlSomers/bpbreid),
[Ultralytics](https://github.com/ultralytics/ultralytics),
[OpenCV 얼굴 탐지·인식](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
