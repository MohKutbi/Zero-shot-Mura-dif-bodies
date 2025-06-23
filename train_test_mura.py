#!/usr/bin/env python
# zero_shot_mura.py
# -------------------------------------------------------------
# Zero-shot bone-fracture classification on MURA v1.1
#   • train ShuffleNet-v2 (5 epochs) once per bone
#   • test the trained network on every bone
#   • per-study metrics with robust label parsing
# -------------------------------------------------------------
import os, sys, copy, contextlib, itertools
from collections import defaultdict, Counter
import warnings

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

# ------------------- CONFIGURATION -------------------
device        = torch.device("cpu")            # set "cuda" if a GPU is available
num_epochs    = 10
batch_size    = 32
learning_rate = 1e-4
num_workers   = 0                              # >0 on Linux/macOS for faster IO

root_dir  = "data"                             # folder holding MURA-v1.1
train_csv = f"{root_dir}/MURA-v1.1/train_labeled_studies.csv"
val_csv   = f"{root_dir}/MURA-v1.1/valid_labeled_studies.csv"  # numeric 0/1

bones = [
    "XR_ELBOW","XR_FINGER","XR_FOREARM",
    "XR_HAND","XR_HUMERUS","XR_SHOULDER","XR_WRIST",
]

out_dir = "runs"
os.makedirs(out_dir, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------- tee helper (console + file) --------------
class Tee(contextlib.AbstractContextManager):
    def __init__(self,*files): self.files=list(files)
    def write(self,txt):  [f.write(txt) for f in self.files]
    def flush(self):      [f.flush()   for f in self.files]
    def __enter__(self):  return self
    def __exit__(self,*exc): self.flush()

# --------------------- DATASET ---------------------
class MURADataset(Dataset):
    """
    samples[i] = (rel_img_path, label_int0/1, study_type, study_id)
    study_id   = "<STUDY_TYPE>/<PATIENT>/<STUDY>"  (globally unique)
    """
    IMG_EXT = {".png",".jpg",".jpeg"}

    def __init__(self, csv_file, root, body_part=None, transform=None):
        self.root, self.transform = root, transform
        self.samples   = []               # (rel_img, label, stype, sid)
        self.cls_count = defaultdict(lambda: {"A":0,"N":0})

        with open(csv_file) as fh:
            for line_num, ln in enumerate(fh,1):
                rel_study, lbl_raw = ln.strip().split(",")

                # --- robust label parse (strings or numeric) ---
                lwr = lbl_raw.lower()
                if lwr in {"positive","1"}:  label = 1
                elif lwr in {"negative","0"}: label = 0
                else:
                    print(f"[WARN] line{line_num}: bad label {lbl_raw}"); continue

                parts = os.path.normpath(rel_study).split(os.sep)
                if len(parts) < 5:                         # XR_ELBOW/.../study
                    continue
                stype = parts[2]
                if body_part and stype != body_part:       # skip other bones
                    continue

                # unique study id
                study_id = os.path.join(parts[2], parts[3], parts[4])
                study_dir= os.path.join(root, rel_study)
                if not os.path.isdir(study_dir): continue

                # recursive image scan
                found=False
                for dp,_,files in os.walk(study_dir):
                    for fn in files:
                        if os.path.splitext(fn.lower())[1] in self.IMG_EXT:
                            found=True
                            rel_img = os.path.join(
                                os.path.relpath(dp, root), fn)
                            self.samples.append((rel_img,label,stype,study_id))
                if not found: continue

                # update per-bone counts
                if label: self.cls_count[stype]["A"] += 1
                else:     self.cls_count[stype]["N"] += 1

        # inverse-freq weights per bone
        self.weights={}
        for st,c in self.cls_count.items():
            A,N,tot=c["A"],c["N"],c["A"]+c["N"]
            self.weights[st]=torch.tensor([
                0.0 if N==0 else tot/(2*N),
                0.0 if A==0 else tot/(2*A)], dtype=torch.float32)

        n_studies=len(set(sid for *_,sid in self.samples))
        print(f"[INFO] {os.path.basename(csv_file)} {body_part or 'ALL'}: "
              f"{len(self.samples)} images, {n_studies} studies")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        rel_img,label,stype,sid=self.samples[idx]
        img=Image.open(os.path.join(self.root,rel_img)).convert("RGB")
        if self.transform: img=self.transform(img)
        return img, torch.tensor(label,dtype=torch.float32), self.weights[stype], sid

# --------------------- TRANSFORMS ------------------
train_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------ TRAIN / TEST -------------------
def train_once_and_test_everywhere(train_bone:str):
    # ------------- LOG SETUP -------------
    train_log = os.path.join(out_dir, f"TRAIN_{train_bone}.txt")
    with open(train_log,"w") as tl, Tee(sys.stdout, tl):

        print(f"\n=== TRAIN on {train_bone} ({num_epochs} epochs) ===", flush=True)
        train_ds=MURADataset(train_csv, root_dir, train_bone, train_tf)
        if len(train_ds)==0: print("No training data.\n"); return
        train_ld=DataLoader(train_ds,batch_size,shuffle=True,num_workers=num_workers)

        # ------------- MODEL --------------
        #net=shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        #net.fc=nn.Linear(net.fc.in_features,1)
        #net.to(device)

        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_ftrs           = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_ftrs, 1)

        crit=nn.BCEWithLogitsLoss(reduction="none")
        opt =optim.Adam(net.parameters(), lr=learning_rate)

        net.train()
        for ep in range(1,num_epochs+1):
            tot_loss=tot_acc=n=0
            for imgs,lbls,wts,_ in train_ld:
                imgs,lbls,wts=imgs.to(device),lbls.unsqueeze(1).to(device),wts.to(device)
                opt.zero_grad()
                out=net(imgs)
                loss=crit(out,lbls)
                # sample weights
                sw=torch.where(lbls==1,wts[:,1],wts[:,0]).unsqueeze(1)
                loss=(loss*sw).mean()
                loss.backward(); opt.step()

                preds=(torch.sigmoid(out)>=0.5).cpu()
                tot_acc+=(preds==lbls.cpu()).sum().item()
                tot_loss+=loss.item()*imgs.size(0); n+=imgs.size(0)
            print(f"Epoch {ep}/{num_epochs} | loss {tot_loss/n:.4f} | acc {tot_acc/n:.4f}", flush=True)

        # freeze for evaluation
        net.eval()

        # ------------- TEST ON ALL BONES -------------
        for test_bone in bones:
            eval_log=os.path.join(out_dir, f"{train_bone}_to_{test_bone}_eff.txt")
            with open(eval_log,"w") as ef, Tee(sys.stdout, ef):
                print(f"\n>>> EVAL: {train_bone} ➜ {test_bone}", flush=True)
                val_ds=MURADataset(val_csv, root_dir, test_bone, val_tf)
                if len(val_ds)==0:
                    print("No validation data."); continue
                val_ld=DataLoader(val_ds,batch_size,False,num_workers=num_workers)

                # per-study aggregation
                probs_by_sid=defaultdict(list)
                lbl_by_sid  = {}
                with torch.no_grad():
                    for imgs,lbls,_,sids in val_ld:
                        pr=torch.sigmoid(net(imgs.to(device))).cpu().numpy().flatten()
                        for p,l,s in zip(pr,lbls.numpy().flatten(),sids):
                            probs_by_sid[s].append(p); lbl_by_sid[s]=int(l)

                preds, lbls=[],[]
                for sid,lbl in lbl_by_sid.items():
                    preds.append(int(np.mean(probs_by_sid[sid])>=0.5))
                    lbls.append(lbl)

                acc = accuracy_score(lbls,preds)
                prec= precision_score(lbls,preds,zero_division=0)
                rec = recall_score(lbls,preds,zero_division=0)

                pos=sum(lbls)
                print(f"Studies: {len(lbls)} | positives: {pos}")
                print(f"Accuracy : {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall   : {rec:.4f}\n")
                print("Classification report:")
                print(classification_report(lbls,preds,digits=4))
                print("Confusion matrix:")
                print(confusion_matrix(lbls,preds),"\n",flush=True)

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    for train_bone in bones:
        train_once_and_test_everywhere(train_bone)
