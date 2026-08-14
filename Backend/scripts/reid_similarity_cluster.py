#!/usr/bin/env python3
"""Huge Re-ID similarity clustering eval on YOLO training images.

1. Detect person crops (YOLOv8n class 0)
2. Embed with PersonReIDEmbedder (MobileNetV3 / OSNet)
3. Cluster by cosine similarity
4. Bucket each crop vs its cluster centroid into score folders:

   results/person_N/images_score_more_than_0.9/
   results/person_N/images_score_more_than_0.7/
   results/person_N/images_score_more_than_0.5/
   results/person_N/images_score_more_than_0.3/
   results/person_N/images_score_below_0.3/

Each image lands in exactly one bucket (highest matching threshold).
Also writes results/summary.json + results/cluster_report.txt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weapon_ai.reid.embeddings import PersonReIDEmbedder, _l2_normalize


SCORE_TIERS = (0.9, 0.7, 0.5, 0.3)


@dataclass
class CropSample:
    path: Path
    split: str
    person_idx: int
    bbox: tuple[int, int, int, int]
    crop_bgr: np.ndarray
    emb: np.ndarray | None = None
    cluster_id: int = -1
    score: float = 0.0


def _iter_images(data_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for split in splits:
        img_dir = data_root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                out.append((split, p))
    return out


def _bucket_name(score: float) -> str:
    for t in SCORE_TIERS:
        if score >= t:
            return f"images_score_more_than_{t}"
    return "images_score_below_0.3"


def _detect_person_crops(
    model,
    image_path: Path,
    *,
    conf: float,
    min_box_px: int,
    max_persons: int,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    bgr = cv2.imread(str(image_path))
    if bgr is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    results = model.predict(
        source=bgr,
        conf=conf,
        classes=[0],
        verbose=False,
        device=0 if _cuda_ok() else "cpu",
    )
    crops: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    if not results:
        return crops
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return crops
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    for i in order[:max_persons]:
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
            continue
        crop = bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crops.append(((x1, y1, x2, y2), crop))
    return crops


_CUDA = None


def _cuda_ok() -> bool:
    global _CUDA
    if _CUDA is None:
        try:
            import torch

            _CUDA = bool(torch.cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA


def _greedy_cluster(embs: np.ndarray, *, link_threshold: float) -> np.ndarray:
    """Assign each sample to best existing cluster if sim >= threshold, else new cluster.

    Prototype = running mean of member embeddings (L2-renormalized).
    """
    n = embs.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    prototypes: list[np.ndarray] = []
    counts: list[int] = []
    # Process high-norm / diverse order: just sequential is fine for eval.
    for i in range(n):
        v = embs[i]
        best_j = -1
        best_sim = -1.0
        for j, proto in enumerate(prototypes):
            sim = float(np.dot(v, proto))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0 and best_sim >= link_threshold:
            labels[i] = best_j
            c = counts[best_j]
            new_mean = _l2_normalize((prototypes[best_j] * c + v) / (c + 1))
            prototypes[best_j] = new_mean
            counts[best_j] = c + 1
        else:
            labels[i] = len(prototypes)
            prototypes.append(v.copy())
            counts.append(1)
    return labels


def _cluster_centroid(embs: np.ndarray, labels: np.ndarray, cid: int) -> np.ndarray:
    mask = labels == cid
    return _l2_normalize(embs[mask].mean(axis=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/insu/Desktop/Model_Training/YOLO/training_data"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
    )
    ap.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits under data-root",
    )
    ap.add_argument("--person-model", type=Path, default=ROOT / "yolov8n.pt")
    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--min-box-px", type=int, default=48)
    ap.add_argument("--max-persons-per-image", type=int, default=6)
    ap.add_argument(
        "--link-threshold",
        type=float,
        default=0.55,
        help="Cosine sim to join an existing person cluster",
    )
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--embed-batch", type=int, default=32)
    ap.add_argument(
        "--reid-backend",
        default="torchreid",
        help="auto|tensorrt|onnx|torchreid|torchvision",
    )
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--max-clusters-export", type=int, default=80)
    ap.add_argument("--clean", action="store_true", help="Wipe out-dir before writing")
    args = ap.parse_args()

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    images = _iter_images(args.data_root, splits)
    if args.max_images and args.max_images > 0:
        images = images[: int(args.max_images)]
    if not images:
        print(f"No images under {args.data_root} splits={splits}")
        return 1

    out_dir: Path = args.out_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"[1/4] Loading person detector {args.person_model}")
    det = YOLO(str(args.person_model))
    print(f"[2/4] Loading Re-ID embedder")
    embedder = PersonReIDEmbedder(device="auto", backend=str(args.reid_backend))
    print(f"       backend={embedder.backend} device={embedder.device} dim={embedder.feature_dim}")
    print(f"[3/4] Detecting + embedding {len(images)} images…")

    samples: list[CropSample] = []
    t0 = time()
    skipped_no_person = 0
    for i, (split, path) in enumerate(images, 1):
        crops = _detect_person_crops(
            det,
            path,
            conf=float(args.person_conf),
            min_box_px=int(args.min_box_px),
            max_persons=int(args.max_persons_per_image),
        )
        if not crops:
            skipped_no_person += 1
        for pi, (bbox, crop) in enumerate(crops):
            samples.append(
                CropSample(
                    path=path,
                    split=split,
                    person_idx=pi,
                    bbox=bbox,
                    crop_bgr=crop,
                )
            )
        if i % 100 == 0 or i == len(images):
            elapsed = time() - t0
            rate = i / max(elapsed, 1e-6)
            print(
                f"  images {i}/{len(images)}  crops={len(samples)}  "
                f"no_person={skipped_no_person}  {rate:.1f} img/s",
                flush=True,
            )

    if not samples:
        print("No person crops found — nothing to cluster.")
        return 1

    print(f"       Embedding {len(samples)} crops…", flush=True)
    t1 = time()
    batch: list[np.ndarray] = []
    batch_idx: list[int] = []

    def flush_batch() -> None:
        nonlocal batch, batch_idx
        if not batch:
            return
        # Sequential embed (model already GPU); keeps API simple / memory bounded.
        for j, crop in zip(batch_idx, batch):
            samples[j].emb = embedder.embed(crop)
        batch = []
        batch_idx = []

    for si, s in enumerate(samples):
        batch.append(s.crop_bgr)
        batch_idx.append(si)
        if len(batch) >= int(args.embed_batch):
            flush_batch()
            if (si + 1) % 500 == 0:
                print(f"  embedded {si + 1}/{len(samples)}", flush=True)
    flush_batch()
    print(f"       embed done in {time() - t1:.1f}s", flush=True)

    embs = np.stack([s.emb for s in samples], axis=0).astype(np.float32)
    print(
        f"[4/4] Clustering {len(samples)} embeddings "
        f"(link_threshold={args.link_threshold})…",
        flush=True,
    )
    labels = _greedy_cluster(embs, link_threshold=float(args.link_threshold))
    for s, lab in zip(samples, labels):
        s.cluster_id = int(lab)

    # Rank clusters by size; remap to person_1..person_K (largest first).
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    ranked = [(int(unique[i]), int(counts[i])) for i in order]
    ranked = [(cid, n) for cid, n in ranked if n >= int(args.min_cluster_size)]
    ranked = ranked[: int(args.max_clusters_export)]
    old_to_person = {cid: (pi + 1) for pi, (cid, _) in enumerate(ranked)}
    keep_old = set(old_to_person.keys())

    # Score vs centroid for kept clusters.
    centroids = {cid: _cluster_centroid(embs, labels, cid) for cid in keep_old}
    for s in samples:
        if s.cluster_id not in centroids:
            s.score = 0.0
            continue
        s.score = float(np.dot(s.emb, centroids[s.cluster_id]))

    # Write crops into buckets.
    bucket_counts: dict[str, dict[str, int]] = {}
    exported = 0
    for s in samples:
        person_num = old_to_person.get(s.cluster_id)
        if person_num is None:
            continue
        person_dir = out_dir / f"person_{person_num}"
        bucket = _bucket_name(s.score)
        dest_dir = person_dir / bucket
        dest_dir.mkdir(parents=True, exist_ok=True)
        score_tag = f"{s.score:.3f}".replace(".", "p")
        stem = s.path.stem
        dest = dest_dir / f"{stem}_p{s.person_idx}_score{score_tag}.jpg"
        # Avoid overwrite collisions.
        if dest.exists():
            dest = dest_dir / f"{stem}_{s.split}_p{s.person_idx}_score{score_tag}.jpg"
        ok = cv2.imwrite(str(dest), s.crop_bgr)
        if ok:
            exported += 1
            key = f"person_{person_num}"
            bucket_counts.setdefault(key, {})
            bucket_counts[key][bucket] = bucket_counts[key].get(bucket, 0) + 1

    # Also dump a small centroid-reference collage note + per-cluster manifest.
    cluster_meta = []
    for cid, n in ranked:
        pnum = old_to_person[cid]
        members = [s for s in samples if s.cluster_id == cid]
        scores = [m.score for m in members]
        meta = {
            "person": pnum,
            "size": n,
            "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "min_score": round(float(np.min(scores)), 4) if scores else 0.0,
            "max_score": round(float(np.max(scores)), 4) if scores else 0.0,
            "buckets": bucket_counts.get(f"person_{pnum}", {}),
            "sample_sources": sorted({m.path.name for m in members})[:12],
        }
        cluster_meta.append(meta)
        manifest = out_dir / f"person_{pnum}" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w") as f:
            json.dump(
                {
                    **meta,
                    "members": [
                        {
                            "source": str(m.path),
                            "split": m.split,
                            "bbox": list(m.bbox),
                            "score": round(m.score, 4),
                            "bucket": _bucket_name(m.score),
                        }
                        for m in sorted(members, key=lambda x: -x.score)
                    ],
                },
                f,
                indent=2,
            )

    summary = {
        "data_root": str(args.data_root),
        "splits": splits,
        "images_scanned": len(images),
        "images_without_person": skipped_no_person,
        "person_crops": len(samples),
        "clusters_total_raw": int(len(unique)),
        "clusters_exported": len(ranked),
        "min_cluster_size": int(args.min_cluster_size),
        "link_threshold": float(args.link_threshold),
        "reid_backend": embedder.backend,
        "reid_device": embedder.device,
        "exported_files": exported,
        "elapsed_s": round(time() - t0, 1),
        "clusters": cluster_meta,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Re-ID similarity clustering report",
        "=" * 40,
        f"images scanned     : {len(images)}",
        f"no-person images   : {skipped_no_person}",
        f"person crops       : {len(samples)}",
        f"raw clusters       : {len(unique)}",
        f"exported persons   : {len(ranked)} (min_size={args.min_cluster_size})",
        f"link threshold     : {args.link_threshold}",
        f"backend            : {embedder.backend} @ {embedder.device}",
        f"exported files     : {exported}",
        f"elapsed            : {summary['elapsed_s']}s",
        "",
        f"{'person':>10} {'size':>6} {'mean':>7} {'min':>7} {'max':>7}  buckets",
    ]
    for m in cluster_meta:
        bucks = ", ".join(f"{k.split('images_')[-1]}={v}" for k, v in sorted(m["buckets"].items()))
        lines.append(
            f"person_{m['person']:<3} {m['size']:>6} {m['mean_score']:>7.3f} "
            f"{m['min_score']:>7.3f} {m['max_score']:>7.3f}  {bucks}"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "cluster_report.txt").write_text(report)
    print(report)
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Results under {out_dir}/person_*/images_score_more_than_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
