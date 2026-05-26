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

# ── Working directory is the repo root ────────────────────
REPO_ROOT = os.path.abspath(".")
print(f">> Repo root: {REPO_ROOT}")

# ── Install DVC first ──────────────────────────────────────
print(">> Installing DVC...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "dvc", "dvc-gdrive", "-q"], check=True)
print(">> DVC installed")

# ── Pull dataset via DVC ───────────────────────────────────
print(">> Pulling dataset via DVC...")
try:
    subprocess.run(["dvc", "pull"], check=True)
    print(">> Dataset ready via DVC")
except Exception as e:
    print(f">> DVC pull failed: {e}")
    print(">> Will download dataset directly as fallback")

# ── Install YOLOv5 ─────────────────────────────────────────
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone",
        "https://github.com/ultralytics/yolov5",
        "--depth", "1"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install",
    "-r", "yolov5/requirements.txt", "-q"], check=True)

# ── Copy yaml files and fix paths inside them ─────────────
print(">> Copying yaml files to yolov5/data/...")
for yaml_file in glob.glob("*.yaml"):
    dest = f"yolov5/data/{yaml_file}"

    # Read yaml content and fix relative paths to absolute
    with open(yaml_file, "r") as f:
        content = f.read()

    # Replace relative path with absolute repo root path
    content = content.replace(
        "path: ../dataset",
        f"path: {REPO_ROOT}/dataset"
    )
    content = content.replace(
        "path: dataset",
        f"path: {REPO_ROOT}/dataset"
    )

    with open(dest, "w") as f:
        f.write(content)

    print(f">> Copied and patched {yaml_file} -> {dest}")

# ── If VOC2007 images missing, download directly ──────────
dataset_yaml = os.path.basename(args["dataset_yaml"])

if "voc" in dataset_yaml.lower():
    voc_path = os.path.join(REPO_ROOT, "dataset", "voc2007", "VOCdevkit")
    images_path = os.path.join(voc_path, "VOC2007", "JPEGImages")
    if not os.path.exists(images_path):
        print(">> VOC2007 images not found, downloading directly...")
        os.makedirs(os.path.join(REPO_ROOT, "dataset", "voc2007"), exist_ok=True)
        subprocess.run([
            "wget", "-q",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "-O", f"{REPO_ROOT}/dataset/voc2007/voc2007.tar"
        ], check=True)
        subprocess.run([
            "tar", "-xf",
            f"{REPO_ROOT}/dataset/voc2007/voc2007.tar",
            "-C", f"{REPO_ROOT}/dataset/voc2007/"
        ], check=True)
        print(">> VOC2007 downloaded and extracted")
    else:
        print(">> VOC2007 images found via DVC")

# ── Build full absolute path for dataset yaml ─────────────
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
