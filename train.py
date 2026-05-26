import os
import sys
import shutil
import glob
import subprocess
from pathlib import Path
from clearml import Task

task = Task.init(
    project_name="YOLOv5-Detection-Demo",
    task_name="baseline"
)

args = {
    "epochs":        10,
    "batch_size":    16,
    "learning_rate": 0.01,
    "weights":       "yolov5s.pt",
    "image_size":    416,
    "dataset_yaml":  "coco128.yaml",
}
args = task.connect(args)

# ── Pull dataset via DVC ───────────────────────────────────
print(">> Pulling dataset via DVC...")
try:
    subprocess.run(["dvc", "pull"], check=True)
    print(">> Dataset ready")
except Exception as e:
    print(f">> DVC pull failed: {e}")

# ── Install YOLOv5 ─────────────────────────────────────────
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone",
        "https://github.com/ultralytics/yolov5",
        "--depth", "1"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install",
    "-r", "yolov5/requirements.txt", "-q"], check=True)

# ── Copy all yaml files into yolov5/data/ ─────────────────
for yaml_file in glob.glob("*.yaml"):
    dest = f"yolov5/data/{yaml_file}"
    shutil.copy(yaml_file, dest)
    print(f">> Copied {yaml_file} -> {dest}")

# ── Always build full absolute path for dataset_yaml ──────
dataset_yaml = os.path.basename(args["dataset_yaml"])
dataset_yaml_full = os.path.abspath(f"yolov5/data/{dataset_yaml}")
print(f">> Using dataset yaml: {dataset_yaml_full}")

if not os.path.exists(dataset_yaml_full):
    raise FileNotFoundError(
        f"dataset yaml not found at: {dataset_yaml_full}\n"
        f"Files in yolov5/data/: {os.listdir('yolov5/data/')}"
    )

# ── Train ──────────────────────────────────────────────────
subprocess.run([
    sys.executable, "yolov5/train.py",
    "--img",     str(args["image_size"]),
    "--batch",   str(args["batch_size"]),
    "--epochs",  str(args["epochs"]),
    "--data",    dataset_yaml_full,
    "--weights", args["weights"],
    "--name",    "run1",
    "--exist-ok",
], check=True)

# ── Save model ─────────────────────────────────────────────
best = Path("yolov5/runs/train/run1/weights/best.pt")
if best.exists():
    task.upload_artifact("best_model", artifact_object=str(best))
    print(">> best.pt saved to ClearML Model Registry")

task.close()
print(">> Done! Check app.clear.ml")
