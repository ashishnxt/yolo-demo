# train.py
# Single training script — all config comes from ClearML dashboard

import os
import sys
import subprocess
from pathlib import Path
from clearml import Task

# ── Init task — gets all config from ClearML dashboard ────
task = Task.init(
    project_name="YOLOv5-Detection-Demo",
    task_name="baseline"
)

# Get hyperparameters set in ClearML dashboard
args = {
    "epochs":         10,
    "batch_size":     16,
    "learning_rate":  0.01,
    "weights":        "yolov5s.pt",
    "image_size":     416,
    "dataset_yaml":   "yolov5/data/coco128.yaml",
}
args = task.connect(args)  # dashboard values override these defaults

# ── Pull dataset via DVC ───────────────────────────────────
print(">> Pulling dataset via DVC...")
try:
    subprocess.run(["dvc", "pull"], check=True)
    print(">> Dataset ready")
except Exception as e:
    print(f">> DVC pull failed: {e}")
    print(">> Downloading COCO128 directly as fallback...")
    os.makedirs("dataset", exist_ok=True)
    subprocess.run(["wget", "-q",
        "https://ultralytics.com/assets/coco128.zip",
        "-O", "dataset/coco128.zip"], check=True)
    subprocess.run(["unzip", "-q", "dataset/coco128.zip",
        "-d", "dataset/"], check=True)

# ── Install YOLOv5 ─────────────────────────────────────────
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone",
        "https://github.com/ultralytics/yolov5",
        "--depth", "1"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install",
    "-r", "yolov5/requirements.txt", "-q"], check=True)

# Copy custom yaml files to yolov5/data/ so YOLOv5 can find them
import shutil, glob
for yaml_file in glob.glob("*.yaml"):
    shutil.copy(yaml_file, f"yolov5/data/{yaml_file}")
    print(f">> Copied {yaml_file} to yolov5/data/")
# ── Train ──────────────────────────────────────────────────
subprocess.run([
    sys.executable, "yolov5/train.py",
    "--img",     str(args["image_size"]),
    "--batch",   str(args["batch_size"]),
    "--epochs",  str(args["epochs"]),
    "--data",    args["dataset_yaml"],
    "--weights", args["weights"],
    "--name",    "run1",
    "--exist-ok",
], check=True)

# ── Save best model to ClearML ─────────────────────────────
best = Path("yolov5/runs/train/run1/weights/best.pt")
if best.exists():
    task.upload_artifact("best_model", artifact_object=str(best))
    print(">> best.pt saved to ClearML Model Registry")

task.close()
print(">> Done! Check app.clear.ml")
