# Classroom Vision · 발표·시연 배포본

**실시간 학생 추적 · ID 라벨 · 인원 집계 · 자리 비움 확인**

2026 한양대학교 ERICA **AI+X 공학융합프로젝트**의 로컬 실행본입니다.
Windows와 Mac에서 사용하며, 서버 연결·학습·테스트·실험 기록 코드는 포함하지 않습니다.

## 실행은 노트북 하나로

1. VS Code에서 **[START.ipynb](START.ipynb)**를 엽니다.
2. **로컬 Python 3.13 커널**을 선택하고 **전체 실행**을 누릅니다.
3. 실행 패널에서 **카메라 / 영상**, **처리 장치**를 고른 뒤 **실행**을 누릅니다.

별도 시연 창이 열립니다. 종료는 노트북의 **중지** 또는 시연 창의 **Q / Esc**입니다.
Python은 [공식 사이트](https://www.python.org/downloads/)에서 설치할 수 있습니다.
VS Code의 Python·Jupyter 확장이 필요합니다.

| 항목 | 동작 |
| --- | --- |
| 플랫폼 | Windows / Mac 자동 인식 |
| 자동 장치 | CUDA → Mac MPS → CPU 순서로 사용 가능 장치 선택 |
| 직접 선택 | 노트북에서 사용 가능한 CUDA·MPS·CPU 선택 |
| 카메라 | 번호 선택 · 기본 0 |
| 영상 | 파일 경로 입력 · HDR MOV 색상 자동 변환 |
| 탐지 크기 | 정확도 우선 960 / 속도 우선 640 |
| 화면 | 이름 지정, ID 초기화, 현재·누적 인원, 자리 비움 경고 |

처음 전체 실행할 때 필요한 패키지와 보조 모델을 준비합니다.
**통합 v2 파인튜닝 가중치가 동봉돼 기본으로 적용됩니다.** 보조 모델까지 약 600MB이며,
기존 모델 캐시가 있으면 재사용합니다. 파일은 SHA-256으로 확인합니다.

## 현장에 가져갈 것

**`deploy` 폴더와 준비된 `models/` 전체**를 가져갑니다.
패키지·모델 준비를 완료한 PC에서는 이후 인터넷 없이 실행할 수 있습니다.
노트북의 **환경 확인**으로 카메라를 켜지 않고 모델 추론을 점검할 수 있습니다.
다른 PC로 옮기면 그 PC의 Python 환경에서 최초 준비를 다시 진행하세요.

Mac은 실행한 VS Code·터미널의 카메라 접근을 허용해야 합니다.
NVIDIA GPU가 장치 목록에 없으면 [CUDA용 PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에
따라 GPU 드라이버와 맞는 PyTorch를 설치하고 커널을 재시작하세요.
영상은 기본 최대 8 FPS를 골라 처리하며 실제 재생 속도는 하드웨어에 따라 달라집니다.

<details>
<summary>터미널에서 직접 실행하는 경우</summary>

```bash
cd deploy
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python models.py --prepare
python app.py --camera 0 --offline
# 영상: python app.py --video "/path/to/demo.MOV" --offline
# NVIDIA: python app.py --camera 0 --device cuda:0 --offline
```

Windows는 `py -3.13 -m venv .venv` 후 `.venv\Scripts\Activate.ps1`로 활성화합니다.

</details>

검증 환경: macOS · Python 3.13 · MPS / CPU. Windows·CUDA 선택 분기는 제공하며,
해당 실기기에서의 실행은 별도로 확인해야 합니다.

## 구현 범위와 출처

사람 탐지, 머리·어깨 검증, 외형·부위·얼굴 비교, 세션 ID와 자리 비움 확인을 구현했습니다.
출석·출튀의 최종 판정과 시험 부정행위 판정은 구현돼 있지 않습니다.
모든 각도·복장 변화·장기 재입장의 ID 유지 정확도도 보장하지 않습니다.
모델·데이터 출처와 라이선스는 **[NOTICE.md](NOTICE.md)**에 정리했습니다.

**팀 어딜가조** · 조영은(PM·디자인·PPT), 이주은(데이터·발표),
손동열(모델·시스템), 목정빈(발표 기획·PPT), 강슬기(데이터)
