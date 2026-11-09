# OpenCV Zoo face models

Face detection and recognition are called through OpenCV's public API; no
upstream Python implementation is copied into `../face_encoder.py`.

The model downloads are pinned to OpenCV Zoo commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7`:
https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models

- YuNet `face_detection_yunet_2026may.onnx` (dynamic input dimensions) is
  licensed under the MIT License; see `LICENSE.yunet`.
  SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
- SFace `face_recognition_sface_2021dec.onnx` (MobileFaceNet trained with SFace)
  is licensed under Apache License 2.0; see `LICENSE.sface`.
  SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

The 2026 YuNet export uses the 2023 model's weights with symbolic height/width,
which supports OpenCV 5's dynamic inference engine and OpenCV 4.13.
Model output is used only as optional evidence. Small, blurred, clipped or
ambiguously assigned faces are skipped; body appearance remains available.
