# AIM-LTCC 학습 준비

현재 상태: 준비 공간. 이 폴더에는 학습 실행기가 아직 없습니다.
기존 `person_tracking/person_tracking.py`의 AIMBackbone/BodyEncoder는 추론 구현입니다.

- `../data/reid/`에 이미지 경로, person_id, session_id, camera_id, split을 가진 CSV와 크롭을 준비합니다.
- 같은 사람의 다른 자세·각도·시간대와, 비슷한 옷을 입은 다른 사람을 포함합니다.
- 옷 변경 대응을 평가하려면 동일 인물의 다른 복장 데이터도 필요합니다.
- 일반화 평가의 train/test 인물은 분리하고, test에는 query/gallery 구분을 둡니다.
- 현재 추론 백본에는 학습용 ID 분류 헤드와 loss/optimizer/data loader가 없습니다.
  원본 학습 코드를 검토해 연결하거나 해당 구성요소를 구현해야 합니다.
- 출력 state dict가 현재 백본의 키·4096차원 특징·정규화와 호환되는지 확인합니다.
- 검색 Rank-1/mAP와 앱의 재입장 ID 복구율을 구분해 기록합니다.

추론 호환 구현: `../../person_tracking/person_tracking.py`의 AIMBackbone.
원본: https://github.com/BoomShakaY/AIM-CCReID
