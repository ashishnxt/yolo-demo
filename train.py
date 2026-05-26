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
    "batch_size":    32,
    "learning_rate": 0.001,
    "weights":       "yolov5m.pt",
    "image_size":    416,
    "dataset_yaml":  "voc2007.yaml",
}
args = task.connect(args)

print(f">> dataset_yaml from dashboard: {args['dataset_yaml']}")

REPO_ROOT = os.path.abspath(".")
print(f">> Repo root: {REPO_ROOT}")

# ── Install DVC ────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install",
    "dvc", "dvc-gdrive", "-q"], check=True)

# ── Pull dataset via DVC ───────────────────────────────────
try:
    subprocess.run(["dvc", "pull"], check=True)
    print(">> Dataset ready via DVC")
except Exception as e:
    print(f">> DVC pull failed: {e}")

# ── Install YOLOv5 ─────────────────────────────────────────
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone",
        "https://github.com/ultralytics/yolov5",
        "--depth", "1"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install",
    "-r", "yolov5/requirements.txt", "-q"], check=True)

# ── Handle VOC2007 — use YOLOv5 official script ───────────
dataset_yaml = os.path.basename(args["dataset_yaml"])

if "voc" in dataset_yaml.lower():
    # YOLOv5 official VOC script downloads + converts to correct structure
    # It creates: datasets/VOC/images/train/ and datasets/VOC/labels/train/
    voc_images = os.path.join(REPO_ROOT, "datasets", "VOC", "images", "train")

    if not os.path.exists(voc_images):
        print(">> Downloading VOC via YOLOv5 official script...")
        subprocess.run([
            sys.executable,
            "yolov5/data/scripts/get_voc.sh"
        ], check=False)  # try shell script first

        # If shell script not available, use Python download
        if not os.path.exists(voc_images):
            print(">> Using Python download fallback...")
            subprocess.run([
                sys.executable, "-c",
                f"""
import os, subprocess
os.chdir("{REPO_ROOT}")
subprocess.run(["bash", "yolov5/data/scripts/get_voc.sh"])
"""
            ], check=False)

    # Use the built-in VOC.yaml that comes with YOLOv5
    # It already points to the correct folder structure
    dataset_yaml = "VOC.yaml"
    print(f">> Using YOLOv5 built-in VOC.yaml")

# ── Copy and patch any custom yaml files ──────────────────
for yaml_file in glob.glob("*.yaml"):
    dest = f"yolov5/data/{yaml_file}"
    with open(yaml_file, "r") as f:
        content = f.read()
    content = content.replace(
        "path: dataset",
        f"path: {REPO_ROOT}/dataset"
    )
    with open(dest, "w") as f:
        f.write(content)

# ── Build full absolute path ───────────────────────────────
dataset_yaml_full = os.path.abspath(f"yolov5/data/{dataset_yaml}")
print(f">> Using dataset yaml: {dataset_yaml_full}")

if not os.path.exists(dataset_yaml_full):
    raise FileNotFoundError(f"yaml not found: {dataset_yaml_full}")

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
