# Robust 3D Object Detection in Adverse Weather (CARLA + PointPillars)

This repository contains the CARLA data generation and evaluation code used in my M.Sc. thesis on robust 3D object detection in adverse weather. It generates KITTI-style data (`image_2`, `velodyne`, `label_2`, `calib`) for different weather and lighting conditions, and evaluates a pre-trained PointPillars detector using OpenPCDet.

---

## 1. Requirements

- Python 3.8+
- [CARLA](https://carla.org/) simulator
  - Tested with Town03
  - LiDAR range ~80 m
- CARLA Python API available on `PYTHONPATH`
- Python packages:
  - `numpy`
  - `Pillow`
  - (optional) `opencv-python` for visualisation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> Note: CARLA’s Python egg is usually installed separately from this project. Please follow the CARLA documentation to install and export the correct `PYTHONPATH`.

---

## 2. CARLA → KITTI Data Generation

The `carla_kitti` package contains a script to generate KITTI-style datasets from CARLA under different weather and lighting conditions.

### 2.1 Available weather presets

Weather presets are defined in `carla_kitti/weather_presets.py`:

- `clear_day`
- `dense_fog`
- `clear_night`
- `rainy_night`
- `heavy_rain_day`

You can use any of these values with the `--weather` argument.

### 2.2 Running the data generator

Start the CARLA server (example on port 2000):

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000
# or CarlaUE4.exe on Windows with the same port
```

From this repo root, generate a scene, e.g. dense fog:

```bash
python -m carla_kitti.generate_kitti_dataset \
  --scene-name scene_dense_fog_v1 \
  --weather dense_fog \
  --frames 600
```

For a clear day baseline:

```bash
python -m carla_kitti.generate_kitti_dataset \
  --scene-name scene_clear_day_v1 \
  --weather clear_day \
  --frames 600
```

### 2.3 Output structure

Each run creates a KITTI-style directory under `output/<scene-name>/`:

```text
output/<scene-name>/
  image_2/    # RGB images (.png)
  velodyne/   # LiDAR point clouds (.bin)
  label_2/    # KITTI-format 3D annotations (.txt)
  calib/      # calibration files (.txt, including P2 and Tr_velo_to_cam)
```

Example:

```text
output/scene_dense_fog_v1/
  image_2/000000.png, 000001.png, ...
  velodyne/000000.bin, 000001.bin, ...
  label_2/000000.txt, 000001.txt, ...
  calib/000000.txt, 000001.txt, ...
```

---

## 3. OpenPCDet Evaluation (PointPillars)

This project uses [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to evaluate a PointPillars 3D detector on the CARLA-generated KITTI-style datasets.

### 3.1 Prepare OpenPCDet

Clone and install OpenPCDet following their README:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
# install dependencies and compile CUDA extensions as described there
```

Symlink or copy one of your CARLA scenes into `OpenPCDet/data`, for example:

```bash
# from OpenPCDet root
ln -s /path/to/robust-3d-detection-carla/output/scene_dense_fog_v1 \
      data/carla_kitti_scene
```

### 3.2 Copy configs

This repository includes custom dataset and model configs for OpenPCDet in:

- `evaluation/configs_openpcdet/carla_kitti_scene.yaml`
- `evaluation/configs_openpcdet/pointpillar_carla_scene.yaml`

Copy them into the appropriate folders inside OpenPCDet:

```bash
# from OpenPCDet root
cp /path/to/robust-3d-detection-carla/evaluation/configs_openpcdet/carla_kitti_scene.yaml \
   cfgs/dataset_configs/

cp /path/to/robust-3d-detection-carla/evaluation/configs_openpcdet/pointpillar_carla_scene.yaml \
   cfgs/kitti_models/
```

Then edit `carla_kitti_scene.yaml` and set `DATA_PATH` to your linked scene:

```yaml
DATASET: "KittiDataset"
DATA_PATH: "/absolute/path/to/OpenPCDet/data/carla_kitti_scene"
```

To evaluate a different CARLA scene (e.g. `scene_clear_day_v1` vs `scene_dense_fog_v1`), update `DATA_PATH` to point to the relevant folder under `OpenPCDet/data`.

### 3.3 Run evaluation

From `OpenPCDet` root, run:

```bash
python tools/test.py \
  --cfg_file cfgs/kitti_models/pointpillar_carla_scene.yaml \
  --ckpt /path/to/pretrained_pointpillars.pth \
  --batch_size 4 \
  --save_to_file
```

This will generate KITTI-format prediction files under a directory like:

```text
output/kitti_models/pointpillar_carla_scene/default/eval/epoch_xx/val/final_result/data/
  000000.txt
  000001.txt
  ...
```

---

## 4. Visualising Predictions on CARLA Images

The script `evaluation/vis/vis_pred_on_image.py` overlays OpenPCDet predictions on the original CARLA images. It supports:

- 2D bounding boxes (KITTI format)
- Optional 3D projected boxes (using `P2` from `calib`)

Example usage:

```bash
python evaluation/vis/vis_pred_on_image.py \
  --images_dir /path/to/robust-3d-detection-carla/output/scene_dense_fog_v1/image_2 \
  --calib_dir  /path/to/robust-3d-detection-carla/output/scene_dense_fog_v1/calib \
  --pred_dir   /path/to/OpenPCDet/output/kitti_models/pointpillar_carla_scene/default/eval/epoch_xx/val/final_result/data \
  --out_dir    /path/to/overlays/scene_dense_fog_v1 \
  --draw3d \
  --score_thresh 0.3
```

This will save overlay images (2D + optional 3D boxes) into `--out_dir`:

```text
/path/to/overlays/scene_dense_fog_v1/
  000000.png
  000001.png
  ...
```

---

## 5. Notes

- This code is research-oriented and was developed for a specific M.Sc. thesis on robustness and hallucination analysis in LiDAR-based 3D object detection.
- The OpenPCDet project is not included in this repository. Please refer to the official OpenPCDet repo for installation, licensing, and full documentation.
- CARLA, OpenPCDet, and KITTI are separate projects; this repository only provides glue code and configs to connect them for robustness experiments in adverse weather.
