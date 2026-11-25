# 출처와 이용 조건

## 프로젝트

2026 한양대학교 ERICA AI+X 공학융합프로젝트 · 팀 어딜가조.
이 폴더는 원본 프로젝트의 추론·화면 코드를 시연용으로 분리한 실행본입니다.
학습 코드, 테스트, 데이터, 원본 강의실 영상, 개인별 저장 특징은 포함하지 않습니다.

## 동봉된 파인튜닝 가중치

- 파일: `models/upper-body-unified-v2-best.pt`
- 모델: YOLO26n · 단일 `person` 클래스 · 통합 상반신 v2
- SHA-256: `857c7d671b47292bf81bcae961d230f1b7913dc5894abdef8e626ddabff0edf2`
- 원본 학습 기록: [통합 학습 리포트](https://github.com/playlistSDY/26ai-x_classroom-vision/blob/8cc0bc9/README.md)
- 학습 데이터: [reddy-projects-zulke/classroom-dodzk v1](https://universe.roboflow.com/reddy-projects-zulke/classroom-dodzk/dataset/1),
  [chokoppang/classroom-o4dnp](https://universe.roboflow.com/chokoppang/classroom-o4dnp)
- 데이터 출처의 CC BY 4.0 표시와 원 리포트의 정제·분할 기준을 유지합니다.
  데이터 라이선스는 아래 모델·코드의 라이선스를 대체하지 않습니다.

## 모델과 코드

| 구성요소 | 출처 | 이용 조건 안내 |
| --- | --- | --- |
| YOLO26n / YOLO26n Pose | [Ultralytics](https://github.com/ultralytics/ultralytics), [Pose 문서](https://docs.ultralytics.com/tasks/pose/) | [Ultralytics 라이선스](https://www.ultralytics.com/license): AGPL-3.0 또는 별도 라이선스 |
| AIM-LTCC 외형 특징 | [AIM-CCReID](https://github.com/BoomShakaY/AIM-CCReID) | 원 프로젝트의 코드·체크포인트 이용 조건을 따릅니다. |
| BPBreID / HRNet | [BPBreID](https://github.com/VlSomers/bpbreid) | [동봉 라이선스](vendor/LICENSE.bpbreid), [출처·수정 내역](vendor/NOTICE.md). Hippocratic License 3.0의 조건을 유지합니다. |
| YuNet / SFace | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | [얼굴 모델 안내](vendor/NOTICE.face-models.md), MIT / Apache-2.0 라이선스 동봉 |
| Pretendard | [Pretendard](https://github.com/orioncactus/pretendard) | [동봉 폰트 라이선스](assets/fonts/LICENSE.txt), SIL Open Font License |
| 동영상 디코더 | [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) / FFmpeg | 설치 패키지의 FFmpeg 빌드 및 라이선스를 따릅니다. |

보조 모델은 원 출처에서 다운로드합니다. 파일별 URL·SHA-256은
[`models/manifest.json`](models/manifest.json)에 고정했습니다.
프로젝트 자체에 새로운 MIT·Apache 등 일괄 라이선스를 부여하지 않으며,
각 외부 구성요소의 이용 조건은 그대로 적용됩니다.
