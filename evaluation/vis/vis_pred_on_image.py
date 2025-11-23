"""
Visualise OpenPCDet predictions on CARLA KITTI-style images.

Draws 2D boxes (and optional 3D projected boxes) using KITTI-format label files.
"""
#!/usr/bin/env python3
import argparse,  cv2, numpy as np
from pathlib import Path

PALETTE = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 0, 0),
    'Cyclist': (0, 128, 255),
    'DontCare': (128,128,128)
}

def parse_pred_file(pred_path, score_thresh=0.3):
    preds = []
    if not pred_path.exists():
        return preds
    for line in pred_path.read_text().strip().splitlines():
        f = line.strip().split()
        if len(f) < 15:  # KITTI label format has 15 fields; OpenPCDet adds score at col 2
            continue
        cls = f[0]
        try:
            # OpenPCDet writes: type, truncated, occluded, alpha, bbox(4), h,w,l, x,y,z, ry, score
            # But some versions: type, score, -1, alpha, bbox(4) ...
            # Try to detect score location
            score = float(f[-1])
            if score > 1.0:  # sometimes not present -> fallback
                score = 1.0
        except:
            score = 1.0
        if score < score_thresh: 
            continue

        # try both layouts for 2D bbox indices
        try:
            # standard KITTI order: cls, trunc, occ, alpha, left, top, right, bottom, h,w,l, x,y,z, ry, (maybe score)
            left, top, right, bottom = map(float, f[4:8])
            h, w, l = map(float, f[8:11])
            x, y, z = map(float, f[11:14])
            ry = float(f[14]) if len(f) >= 15 else 0.0
        except:
            # alt layout seen in some dumps: cls, score, ?, alpha, left, top, right, bottom, ...
            left, top, right, bottom = map(float, f[4:8])
            h, w, l = map(float, f[8:11])
            x, y, z = map(float, f[11:14])
            ry = float(f[14]) if len(f) >= 15 else 0.0

        preds.append({
            'cls': cls, 'score': score,
            'bbox': (left, top, right, bottom),
            'dims': (h, w, l),
            'loc': (x, y, z),  # in camera coords
            'ry': ry
        })
    return preds

def cube_corners_cam(h, w, l, x, y, z, ry):
    # KITTI box is bottom-centered at (x,y,z) in camera coords; dims are h,w,l
    # Build corners in object coord, then rotate around Y and translate
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [   0,    0,    0,    0,  -h,  -h,  -h,  -h]  # 0 at bottom
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.vstack([x_corners, y_corners, z_corners])  # (3,8)

    R = np.array([[ np.cos(ry), 0, np.sin(ry)],
                  [          0, 1,          0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners_rot = R @ corners
    corners_rot[0, :] += x
    corners_rot[1, :] += y
    corners_rot[2, :] += z
    return corners_rot  # (3,8)

def project(P2, pts3d):  # pts3d: (3,N)
    pts_h = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])  # (4,N)
    proj = P2 @ pts_h  # (3,N)
    proj[:2, :] /= proj[2:, :]
    return proj[:2, :]  # (2,N)

def draw_3d_box(img, P2, pred, color):
    h, w, l = pred['dims']
    x, y, z = pred['loc']
    ry = pred['ry']
    corners = cube_corners_cam(h, w, l, x, y, z, ry)  # (3,8)
    if (corners[2] <= 0).any():  # behind camera
        return img
    pts2d = project(P2, corners).T.astype(int)  # (8,2)

    # connectivity of 3D bbox corners
    edges = [(0,1),(1,2),(2,3),(3,0),  # bottom
             (4,5),(5,6),(6,7),(7,4),  # top
             (0,4),(1,5),(2,6),(3,7)]  # vertical
    for i,j in edges:
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), color, 2, cv2.LINE_AA)
    return img

def load_P2(calib_path):
    # read P2 from calib file
    P2 = None
    for line in calib_path.read_text().strip().splitlines():
        if line.startswith('P2:'):
            vals = [float(x) for x in line.split(':',1)[1].strip().split()]
            P2 = np.array(vals, dtype=np.float32).reshape(3,4)
            break
    return P2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True, help='path to image_2')
    ap.add_argument('--pred_dir', required=True, help='path to prediction txts (final_result/data)')
    ap.add_argument('--out_dir', required=True, help='where to save overlays')
    ap.add_argument('--calib_dir', default=None, help='if given, draw 3D boxes using P2 from calib')
    ap.add_argument('--score_thresh', type=float, default=0.3)
    ap.add_argument('--draw3d', action='store_true', help='enable 3D projected boxes (needs calib)')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None

    pred_files = sorted(pred_dir.glob('*.txt'))
    print(f"[INFO] Found {len(pred_files)} prediction files")

    for pf in pred_files:
        stem = pf.stem
        img_path = images_dir / f"{stem}.png"
        if not img_path.exists():
            # try jpg
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        preds = parse_pred_file(pf, args.score_thresh)
        P2 = None
        if args.draw3d and calib_dir is not None:
            calib_path = calib_dir / f"{stem}.txt"
            if calib_path.exists():
                P2 = load_P2(calib_path)

        for p in preds:
            color = PALETTE.get(p['cls'], (255,255,255))
            # draw 2D box
            l,t,r,b = map(int, p['bbox'])
            cv2.rectangle(img, (l,t), (r,b), color, 2)
            label = f"{p['cls']} {p['score']:.2f}"
            cv2.putText(img, label, (l, max(10, t-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # optional 3D
            if args.draw3d and (P2 is not None):
                img = draw_3d_box(img, P2, p, color)

        cv2.imwrite(str(out_dir / f"{stem}.png"), img)

if __name__ == '__main__':
    main()
