# BPBreID 학습 준비

현재 상태: 준비 공간. 이 폴더에는 학습 실행기가 아직 없습니다.
기존 `person_tracking/partial_encoder.py`는 학습용 헤드를 제외한 추론 구현입니다.

- `../data/reid/`에 person_id/session_id/camera_id/split과 크롭을 준비합니다.
- 책상 가림, 상체만 보임, 다른 사람과 겹침 사례를 포함합니다.
- 원본 학습 파이프라인에 맞는 신체 부위 파싱 라벨/마스크 또는 의사 라벨 생성 절차가 필요합니다.
- 일반화 평가의 train/test 인물은 분리하고, test에는 query/gallery를 구성합니다.
- 원본 학습 코드의 데이터셋 어댑터·loss·학습 헤드를 연결해야 합니다.
- 현재 앱은 HRNet-W32, 8개 부위, 부위별 512차원 출력을 기대합니다.
  학습한 가중치를 적용하기 전에 state dict와 설정 호환성을 검사합니다.
- ReID 지표 외에 가림 구간의 ID 전환·재입장 복구율을 평가합니다.

원본 학습 절차: https://github.com/VlSomers/bpbreid
기존 출처·라이선스: `../../person_tracking/vendor/NOTICE.md`, `LICENSE.bpbreid`.
