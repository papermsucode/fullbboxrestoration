# fullbboxrestoration
The accompanying code for the work [Darker than Black-Box: Face Reconstruction from Similarity Queries](https://arxiv.org/abs/2106.14290) by Anton Razzhigaev, Klim Kireev, Igor Udovichenko, Aleksandr Petiushko.

**How to recover face with pretrained eigenfaces:**
1. download checkpoints from https://drive.google.com/drive/folders/1g0P32pX8BydZ4YA66H6Aqle1qC8oR3d9?usp=sharing
2. launch "run all cells" recover_face.ipynb

**How to train eigenfaces:**
1. download ClelebA dataset
2. run train_eigenfaces.ipynb

### requirements:
* python 3.8
* torch 1.8
