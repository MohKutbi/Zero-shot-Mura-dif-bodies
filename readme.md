# Zero‑Shot Bone‑Fracture Classification on MURA

Efficient **cross‑bone** transfer learning for radiographs.  This repo lets you train **EfficientNet‑B0** (or any other *torchvision* backbone) on **one** body part from the [MURA v1.1 dataset](https://stanfordmlgroup.github.io/competitions/mura/) and instantly evaluate it on **all seven** body parts – reporting **per‑study** accuracy, precision, recall, confusion matrix and a classification report.

> **Why?** Compare how well a fracture detector trained on one anatomy generalises to another ("zero‑shot" transfer) – a common research question in medical imaging.

---

## Features

| ✔︎                                                                         | Feature |
| -------------------------------------------------------------------------- | ------- |
| Robust label parsing – works with `positive/negative` **or** numeric `1/0` |         |
| Globally‑unique `study_id` → `<BONE>/<PATIENT>/<STUDY>`                    |         |
| Recursively loads **every** image inside each study folder                 |         |
| Handles heavy class imbalance with **per‑bone class weights**              |         |
| Fast: *train once / test many* (default 5 epochs)                          |         |
| Logs progress to console **and** `runs/` folder                            |         |
| Pure PyTorch & Torchvision; CPU by default, GPU optional                   |         |

---

## Quick start

```bash
# clone your fork and enter it
$ git clone https://github.com/<you>/zero‑shot‑mura.git && cd zero‑shot‑mura

# create env (optional)
$ python -m venv venv && source venv/bin/activate  # on Windows: venv\Scripts\activate

# install requirements
$ pip install -r requirements.txt

# download & unzip MURA v1.1 into ./data (folder structure must match release)
$ python zero_shot_mura.py          # CPU run
# or
$ CUDA_VISIBLE_DEVICES=0 python zero_shot_mura.py  # GPU run
```

Logs are saved to `runs/TRAIN_<BONE>.txt` and `<TRAIN>_to_<TEST>.txt`.

---

## Directory layout

```
zero‑shot‑mura/
├── zero_shot_mura.py          # main script (training + evaluation)
├── requirements.txt           # pip deps
├── LICENSE                    # MIT by default
├── runs/                      # logs & best checkpoints (*.pth)
└── data/
    └── MURA-v1.1/             # place the official dataset here
        ├── train_labeled_studies.csv
        ├── valid_labeled_studies.csv  (numeric labels 0/1)
        └── <images & folders>
```

---

## Configuration

Edit the constants at the top of `zero_shot_mura.py` –

```python
num_epochs    = 5          # more epochs ⇒ better accuracy, longer runtime
batch_size    = 32         # increase on GPU if memory allows
learning_rate = 1e-4       # Adam LR
device        = "cuda"     # switch from "cpu" when you have a GPU
```

Change the backbone:

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
```

Any *torchvision* model with a single Linear classifier can be swapped in a similar way.

---

## Results format

Each evaluation log contains e.g.:

```
>>> EVAL: train XR_ELBOW ➜ test XR_HAND
Studies: 167 | positives: 26
Accuracy : 0.5988
Precision: 0.4375
Recall   : 0.6154
...
Confusion matrix:
[[108  33]
 [ 11  15]]
```

Re‑create a cross‑bone accuracy matrix just like Table 2 of the MURA paper.

---

## Citation

If you use this repo in academic work, please cite the original dataset:

```
@article{rajpurkar2017mura,
  title={MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs},
  author={Pranav Rajpurkar and others},
  journal={arXiv:1712.06957},
  year={2017}
}
```

---

## License

This project is released under the **MIT License** – see [LICENSE](LICENSE).

