# BPBreID HRNet inference component

`hrnet.py` is adapted from the official BPBreID repository:
https://github.com/VlSomers/bpbreid/blob/a2dc4304284784d8b2c061764512f0698aa38c21/torchreid/models/hrnet.py

The upstream code attributes the HRNet implementation to:
https://github.com/HRNet/HRNet-Image-Classification
and https://github.com/CASIA-IVA-Lab/ISP-reID

BPBreID authors: Vladimir Somers, Christophe De Vleeschouwer, Alexandre Alahi.
Paper: Body Part-Based Representation Learning for Occluded Person
Re-Identification, WACV 2023, https://arxiv.org/abs/2211.03679

The upstream Hippocratic License 3.0 (LAW-MEDIA-MIL-SOC-SV) is included
in `LICENSE.bpbreid`. This is a conditional license, not MIT or Apache.

Local adaptations replace YACS configuration nodes with identical plain
Python dictionaries. The feature architecture and parameter names are kept
unchanged. The initialization log is suppressed before loading trained weights.
`../partial_encoder.py` implements the published inference-only
attention, weighted pooling and dimensionality reduction from BPBreID.
The original checkpoint stores eight semantic parts, despite example test
YAML files listing five-part preprocessing. The stored trained architecture
is used here: head, left/right arms, torso, left/right legs, left/right feet.
