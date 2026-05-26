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
print(">> Installing DVC...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "dvc", "dvc-gdrive", "-q"], check=True)

# ── Pull dataset via DVC ───────────────────────────────────
print(">> Pulling dataset via DVC...")
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

# ── Handle VOC2007 dataset ─────────────────────────────────
dataset_yaml = os.path.basename(args["dataset_yaml"])

if "voc" in dataset_yaml.lower():
    voc_root    = os.path.join(REPO_ROOT, "dataset", "voc2007", "VOCdevkit", "VOC2007")
    images_path = os.path.join(voc_root, "JPEGImages")
    labels_path = os.path.join(voc_root, "labels")
    annot_path  = os.path.join(voc_root, "Annotations")

    # Download if missing
    if not os.path.exists(images_path):
        print(">> VOC2007 not found, downloading...")
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
        print(">> VOC2007 downloaded")

    # Convert XML annotations to YOLO TXT format if not done yet
    if not os.path.exists(labels_path) or len(os.listdir(labels_path)) == 0:
        print(">> Converting VOC XML annotations to YOLO TXT format...")
        os.makedirs(labels_path, exist_ok=True)

        import xml.etree.ElementTree as ET

        # VOC class names — must match voc2007.yaml order
        VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

        converted = 0
        for xml_file in glob.glob(os.path.join(annot_path, "*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_w = int(root.find("size/width").text)
            img_h = int(root.find("size/height").text)

            txt_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in VOC_CLASSES:
                    continue
                cls_id = VOC_CLASSES.index(cls_name)

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Convert to YOLO format (normalized cx, cy, w, h)
                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h
                w  = (xmax - xmin) / img_w
                h  = (ymax - ymin) / img_h

                txt_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Save txt file with same name as xml
            txt_name = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
            with open(os.path.join(labels_path, txt_name), "w") as f:
                f.write("\n".join(txt_lines))
            converted += 1

        print(f">> Converted {converted} XML files to YOLO TXT format")
    else:
        print(f">> Labels already exist: {len(os.listdir(labels_path))} files")

# ── Copy and patch yaml files ──────────────────────────────
print(">> Copying yaml files to yolov5/data/...")
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
    print(f">> Copied and patched {yaml_file} -> {dest}")

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
