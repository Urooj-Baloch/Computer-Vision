import os
import glob
import json
import pickle
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import numpy as np
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

cv2.setNumThreads(1)
rng = np.random.default_rng(7)
tf.random.set_seed(7)

CLASSES = ["background", "raccoon", "kangaroo"]
COLORS = {1: (0, 200, 0), 2: (255, 120, 0)}
MAX_SIDE = 320
CROP = 64
MAX_PROPOSALS = 400
OUT = "output"
os.makedirs(OUT + "/predictions", exist_ok=True)


def parse_voc(xml_path, image_path):
    root = ET.parse(xml_path).getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASSES:
            continue
        b = obj.find("bndbox")
        boxes.append([float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASSES.index(name))
    return {"path": image_path, "boxes": np.array(boxes, np.float32), "labels": np.array(labels)}


def load_dataset():
    records = []
    for xml_path in sorted(glob.glob("dataset/raccoon/annotations/*.xml")):
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        image_path = f"dataset/raccoon/images/{stem}.jpg"
        if os.path.exists(image_path):
            records.append(parse_voc(xml_path, image_path))
    for xml_path in sorted(glob.glob("dataset/kangaroo/annots/*.xml")):
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        image_path = f"dataset/kangaroo/images/{stem}.jpg"
        if os.path.exists(image_path):
            records.append(parse_voc(xml_path, image_path))
    return [r for r in records if len(r["boxes"]) > 0]


def prepare(record):
    img = cv2.imread(record["path"])
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    sh, sw = small.shape[:2]
    keep = [(x, y, x + rw, y + rh) for x, y, rw, rh in rects[:MAX_PROPOSALS * 2] if rw >= 24 and rh >= 24 and rw * rh < 0.95 * sw * sh]
    props = np.array(keep[:MAX_PROPOSALS], np.float32)
    return {"scale": s, "size": (sh, sw), "props": props, "boxes": record["boxes"] * s}


def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def warp(small, boxes, flip=False):
    sh, sw = small.shape[:2]
    out = np.zeros((len(boxes), CROP, CROP, 3), np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2, y2 = int(min(sw, np.ceil(x2))), int(min(sh, np.ceil(y2)))
        patch = small[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
        patch = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        out[i] = patch[:, ::-1] if flip else patch
    return out


def encode(props, gts):
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    px, py = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    gw, gh = gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1]
    gx, gy = gts[:, 0] + gw / 2, gts[:, 1] + gh / 2
    return np.stack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)], axis=1)


def decode(props, deltas):
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    px, py = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    cx, cy = px + deltas[:, 0] * pw, py + deltas[:, 1] * ph
    w, h = pw * np.exp(np.clip(deltas[:, 2], -1, 1)), ph * np.exp(np.clip(deltas[:, 3], -1, 1))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thr]
    return keep


def build_models(n_classes=3):
    inp = keras.Input((CROP, CROP, 3))
    x = inp
    for ch in (32, 64, 128, 256):
        x = layers.Conv2D(ch, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    emb = layers.Dense(256, activation="relu", name="embed")(x)
    out = layers.Dense(n_classes, activation="softmax")(layers.Dropout(0.4)(emb))
    return keras.Model(inp, out), keras.Model(inp, emb)


def to_tensor(crops):
    return (crops.astype(np.float32) / 255.0 - 0.45) / 0.25


def embed_all(embed_model, crops, batch=512):
    if len(crops) == 0:
        return np.zeros((0, 256), np.float32)
    return embed_model.predict(to_tensor(crops), batch_size=batch, verbose=0)


def average_precision(rec, prec):
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(detections, ground_truth, cls, thr=0.5):
    gt = {k: v["boxes"][v["labels"] == cls] for k, v in ground_truth.items()}
    npos = sum(len(v) for v in gt.values())
    used = {k: np.zeros(len(v), bool) for k, v in gt.items()}
    dets = sorted([(s, k, b) for k, lst in detections.items() for (c, s, b) in lst if c == cls], key=lambda t: -t[0])
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for i, (s, k, b) in enumerate(dets):
        g = gt[k]
        if len(g) == 0:
            fp[i] = 1
            continue
        ious = iou_matrix(np.array([b], np.float32), g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= thr and not used[k][j]:
            tp[i] = 1
            used[k][j] = True
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    return average_precision(rec, prec) if len(dets) else 0.0, rec, prec


def draw(img, dets, gts=None, scale=1.0, min_score=0.4):
    canvas = img.copy()
    thick = max(2, int(round(max(img.shape[:2]) / 300)))
    fs = max(img.shape[:2]) / 900
    if gts is not None:
        for b in gts["boxes"]:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thick)
    for c, s, b in dets:
        if s < min_score:
            continue
        x1, y1, x2, y2 = [int(round(v / scale)) for v in b]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[c][::-1], thick)
        cv2.putText(canvas, f"{CLASSES[c]} {s:.2f}", (x1 + 3, max(y1 + int(22 * fs) + 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, max(fs, 0.5), COLORS[c][::-1], max(1, thick - 1), cv2.LINE_AA)
    return canvas


def main():
    records = load_dataset()
    print("Images:", len(records), "| raccoon boxes:", sum(int((r["labels"] == 1).sum()) for r in records), "| kangaroo boxes:", sum(int((r["labels"] == 2).sum()) for r in records))
    order = rng.permutation(len(records))
    n_train = int(0.8 * len(records))
    train_ids, test_ids = order[:n_train], order[n_train:]
    print("Train images:", len(train_ids), "| test images:", len(test_ids))

    cache = OUT + "/proposals.pkl"
    if os.path.exists(cache):
        prepared = pickle.load(open(cache, "rb"))
    else:
        t0 = time.time()
        with Pool(2) as pool:
            prepared = pool.map(prepare, records, chunksize=4)
        pickle.dump(prepared, open(cache, "wb"))
        print(f"Selective search on {len(records)} images took {time.time() - t0:.0f}s")
    per_image_props = np.mean([len(p["props"]) for p in prepared])
    print(f"Average proposals per image: {per_image_props:.0f}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, i in zip(axes.ravel(), list(train_ids[:4]) + list(train_ids[-4:])):
        img = cv2.imread(records[i]["path"])
        for b, l in zip(records[i]["boxes"], records[i]["labels"]):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[l][::-1], 3)
            cv2.putText(img, CLASSES[l], (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[l][::-1], 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(os.path.basename(records[i]["path"]))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT + "/fig1_ground_truth.png", dpi=80)
    plt.close()

    sample = prepared[train_ids[0]]
    small = cv2.resize(cv2.imread(records[train_ids[0]]["path"]), (sample["size"][1], sample["size"][0]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, n in zip(axes, [50, 150, len(sample["props"])]):
        canvas = small.copy()
        for x1, y1, x2, y2 in sample["props"][:n]:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Selective search: {n} region proposals")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT + "/fig2_selective_search.png", dpi=80)
    plt.close()

    crops, labels, flips = [], [], []
    deltas_src = []
    for i in train_ids:
        p, rec = prepared[i], records[i]
        img = cv2.resize(cv2.imread(rec["path"]), (p["size"][1], p["size"][0]), interpolation=cv2.INTER_AREA)
        cand = np.concatenate([p["props"], p["boxes"]]) if len(p["props"]) else p["boxes"]
        ious = iou_matrix(cand, p["boxes"])
        best, arg = ious.max(1), ious.argmax(1)
        pos = np.where(best >= 0.5)[0]
        neg = np.where(best < 0.3)[0]
        if len(pos) > 24:
            pos = rng.choice(pos, 24, replace=False)
        n_neg = min(len(neg), max(3 * len(pos), 24))
        neg = rng.choice(neg, n_neg, replace=False) if len(neg) else neg
        for idx in pos:
            lbl = rec["labels"][arg[idx]]
            crops.append(warp(img, cand[idx:idx + 1])[0])
            labels.append(lbl)
        for idx in neg:
            crops.append(warp(img, cand[idx:idx + 1])[0])
            labels.append(0)
    crops = np.array(crops)
    labels = np.array(labels)
    print("CNN training crops:", len(crops), "| per class:", {CLASSES[c]: int((labels == c).sum()) for c in range(3)})

    perm = rng.permutation(len(crops))
    n_val = int(0.1 * len(crops))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model_path = OUT + "/region_cnn.keras"
    history = {"loss": [], "train_acc": [], "val_acc": []}
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        embed_model = keras.Model(model.input, model.get_layer("embed").output)
        print("Loaded trained region CNN from", model_path)
    else:
        counts = np.bincount(labels, minlength=3)
        class_weight = {c: float(len(labels) / (3 * counts[c])) for c in range(3)}
        model, embed_model = build_models()
        steps = 14 * int(np.ceil(len(tr_idx) / 128))
        schedule = keras.optimizers.schedules.CosineDecay(2e-3, steps)
        model.compile(optimizer=keras.optimizers.Adam(schedule, weight_decay=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = {"loss": [], "train_acc": [], "val_acc": []}
        x_val = to_tensor(crops[val_idx])
        y_val = labels[val_idx]
        for epoch in range(14):
            xb = crops[tr_idx].copy()
            flip = rng.random(len(xb)) < 0.5
            xb[flip] = xb[flip][:, :, ::-1]
            x = to_tensor(np.ascontiguousarray(xb))
            x = x * (1 + 0.15 * rng.standard_normal((len(x), 1, 1, 1)).astype(np.float32)) + 0.15 * rng.standard_normal((len(x), 1, 1, 1)).astype(np.float32)
            h = model.fit(x, labels[tr_idx], batch_size=128, epochs=1, class_weight=class_weight, shuffle=True, verbose=0)
            val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=512, verbose=0)
            history["loss"].append(float(h.history["loss"][0]))
            history["train_acc"].append(float(h.history["accuracy"][0]))
            history["val_acc"].append(float(val_acc))
            print(f"epoch {epoch + 1:2d}  loss {history['loss'][-1]:.4f}  train acc {history['train_acc'][-1]:.3f}  val acc {val_acc:.3f}")
        model.save(OUT + "/region_cnn.keras")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if history["loss"]:
        json.dump(history, open(OUT + "/history.json", "w"))
    elif os.path.exists(OUT + "/history.json"):
        history = json.load(open(OUT + "/history.json"))
    axes[0].plot(history["loss"], marker="o")
    axes[0].set_title("Region CNN training loss")
    axes[0].set_xlabel("epoch")
    axes[1].plot(history["train_acc"], marker="o", label="train")
    axes[1].plot(history["val_acc"], marker="o", label="validation")
    axes[1].set_title("Region classification accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT + "/fig3_training_curves.png", dpi=90)
    plt.close()

    feats = embed_all(embed_model, crops)
    svms = {}
    for c in (1, 2):
        svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)
        svm.fit(feats, (labels == c).astype(int))
        svms[c] = svm
    print("Trained one linear SVM per class on CNN features")

    reg_x, reg_y = {1: [], 2: []}, {1: [], 2: []}
    for i in train_ids:
        p, rec = prepared[i], records[i]
        if not len(p["props"]):
            continue
        img = cv2.resize(cv2.imread(rec["path"]), (p["size"][1], p["size"][0]), interpolation=cv2.INTER_AREA)
        ious = iou_matrix(p["props"], p["boxes"])
        best, arg = ious.max(1), ious.argmax(1)
        for c in (1, 2):
            sel = np.where((best >= 0.6) & (rec["labels"][arg] == c))[0]
            if len(sel) == 0:
                continue
            f = embed_all(embed_model, warp(img, p["props"][sel]))
            reg_x[c].append(f)
            reg_y[c].append(encode(p["props"][sel], p["boxes"][arg[sel]]))
    regressors = {}
    for c in (1, 2):
        regressors[c] = Ridge(alpha=1.0).fit(np.concatenate(reg_x[c]), np.concatenate(reg_y[c]))
    print("Trained bounding-box regressors:", {CLASSES[c]: len(np.concatenate(reg_x[c])) for c in (1, 2)})

    detections_reg, detections_raw, ground_truth, times = {}, {}, {}, []
    pred_images = {}
    for i in test_ids:
        p, rec = prepared[i], records[i]
        key = os.path.basename(rec["path"])
        ground_truth[key] = {"boxes": p["boxes"], "labels": rec["labels"]}
        t0 = time.time()
        full = cv2.imread(rec["path"])
        small = cv2.resize(full, (p["size"][1], p["size"][0]), interpolation=cv2.INTER_AREA)
        props = prepare(rec)["props"]
        t_prop = time.time() - t0
        f = embed_all(embed_model, warp(small, props))
        det_reg, det_raw = [], []
        for c in (1, 2):
            scores = svms[c].decision_function(f)
            sel = np.where(scores > 0.0)[0]
            if len(sel) == 0:
                continue
            raw_boxes = props[sel]
            reg_boxes = decode(props[sel], regressors[c].predict(f[sel]))
            reg_boxes[:, [0, 2]] = np.clip(reg_boxes[:, [0, 2]], 0, p["size"][1] - 1)
            reg_boxes[:, [1, 3]] = np.clip(reg_boxes[:, [1, 3]], 0, p["size"][0] - 1)
            for src, dest in ((raw_boxes, det_raw), (reg_boxes, det_reg)):
                keep = nms(src, scores[sel], 0.3)
                dest += [(c, float(scores[sel][k]), src[k]) for k in keep]
        times.append((t_prop, time.time() - t0))
        detections_reg[key] = det_reg
        detections_raw[key] = det_raw
        pred_images[key] = (full, det_reg, p["scale"], rec)
    print(f"Per-image inference: proposals {np.mean([t[0] for t in times]):.2f}s, total {np.mean([t[1] for t in times]):.2f}s")

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in (1, 2):
        ap_reg, rec_r, prec_r = evaluate(detections_reg, ground_truth, c)
        ap_raw, rec_w, prec_w = evaluate(detections_raw, ground_truth, c)
        results[CLASSES[c]] = {"AP50_with_bbox_regression": ap_reg, "AP50_without_bbox_regression": ap_raw}
        axes[c - 1].plot(rec_w, prec_w, label=f"raw proposals (AP {ap_raw:.3f})")
        axes[c - 1].plot(rec_r, prec_r, label=f"with regression (AP {ap_reg:.3f})")
        axes[c - 1].set_title(f"Precision-recall: {CLASSES[c]}")
        axes[c - 1].set_xlabel("recall")
        axes[c - 1].set_ylabel("precision")
        axes[c - 1].set_xlim(0, 1)
        axes[c - 1].set_ylim(0, 1.02)
        axes[c - 1].legend()
    plt.tight_layout()
    plt.savefig(OUT + "/fig5_precision_recall.png", dpi=90)
    plt.close()
    results["mAP50_with_bbox_regression"] = float(np.mean([results[CLASSES[c]]["AP50_with_bbox_regression"] for c in (1, 2)]))
    results["mAP50_without_bbox_regression"] = float(np.mean([results[CLASSES[c]]["AP50_without_bbox_regression"] for c in (1, 2)]))
    results["avg_seconds_per_image_total"] = float(np.mean([t[1] for t in times]))
    results["avg_seconds_per_image_proposals"] = float(np.mean([t[0] for t in times]))
    results["avg_proposals_per_image"] = float(per_image_props)
    results["train_images"] = int(len(train_ids))
    results["test_images"] = int(len(test_ids))
    json.dump(results, open(OUT + "/metrics.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

    keys = list(pred_images)
    for key in keys:
        full, det, s, rec = pred_images[key]
        cv2.imwrite(f"{OUT}/predictions/{key}", draw(full, det, {"boxes": rec["boxes"]}, s))
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax, key in zip(axes.ravel(), keys[:12]):
        full, det, s, rec = pred_images[key]
        ax.imshow(cv2.cvtColor(draw(full, det, {"boxes": rec["boxes"]}, s), cv2.COLOR_BGR2RGB))
        ax.set_title(key)
        ax.axis("off")
    plt.suptitle("R-CNN test detections (white = ground truth)", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT + "/fig4_test_detections.png", dpi=70)
    plt.close()

    rows = [("R-CNN (this lab)", results["avg_seconds_per_image_total"], per_image_props)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Proposals (selective search)", "CNN + SVM + regression"], [results["avg_seconds_per_image_proposals"], results["avg_seconds_per_image_total"] - results["avg_seconds_per_image_proposals"]], color=["tab:orange", "tab:blue"])
    ax.set_ylabel("seconds per image")
    ax.set_title("R-CNN inference time breakdown")
    plt.tight_layout()
    plt.savefig(OUT + "/fig6_inference_time.png", dpi=90)
    plt.close()


if __name__ == "__main__":
    main()
