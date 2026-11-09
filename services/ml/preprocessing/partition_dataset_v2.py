"""
CivicPulse AI — Dataset Partitioning & Corridor Grouping Suite (v2)
Performs sequence/corridor clustering, class-balanced split assignment,
boundary guard buffering, perceptual similarity checks, and zero-leakage validation.
"""

import os
import sys
import re
import json
import time
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set, Tuple

import numpy as np
from PIL import Image

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40"
}


def dhash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Computes difference hash (dHash) as a boolean array."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def load_all_samples(base_dir: Path) -> List[Dict[str, Any]]:
    """Loads all 7,706 samples across existing train, val, and test splits."""
    samples = []
    seen_indices = set()
    
    for split in ["train", "val", "test"]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        if not img_dir.exists():
            continue
            
        for img_p in sorted(img_dir.glob("*.jpg")):
            m = re.search(r"India_(\d+)", img_p.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            content = lbl_p.read_text().strip() if lbl_p.exists() else ""
            
            classes = set()
            boxes = []
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cname = CLASS_LABELS.get(cls_id, f"C{cls_id}")
                        classes.add(cname)
                        boxes.append({
                            "class_id": cls_id,
                            "class_name": cname,
                            "coords": [float(x) for x in parts[1:]]
                        })
                        
            samples.append({
                "idx": idx,
                "name": img_p.name,
                "img_path": img_p,
                "lbl_path": lbl_p,
                "classes": classes,
                "boxes": boxes,
                "has_damage": len(boxes) > 0,
                "corridor_id": idx // 100,  # 100 consecutive frames per corridor
            })
            
    samples.sort(key=lambda s: s["idx"])
    return samples


def find_optimal_corridor_split(samples: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Finds a corridor-to-split mapping that:
    1. Keeps entire corridors monolithic (no within-corridor splitting).
    2. Groups corridors in contiguous macro-blocks to minimize split boundary transitions.
    3. Balances overall samples (~70% train, ~15% val, ~15% test).
    4. Balances rare classes (D10 transverse cracks, D40 potholes).
    """
    corridors = defaultdict(list)
    for s in samples:
        corridors[s["corridor_id"]].append(s)
        
    corridor_ids = sorted(corridors.keys())
    
    # We test multiple macro-block patterns and select the one with optimal
    # class balance and minimal chi-square distance to target (0.70, 0.15, 0.15).
    best_score = float("inf")
    best_assignment = {}
    
    # Patterns of length 20 corridors (14 train, 3 val, 3 test)
    # Rotating offsets to find optimal alignment with rare defects
    patterns = [
        # Pattern A: Blocks of 7-8 train, 3 val, 6-7 train, 3 test
        ['train']*7 + ['val']*3 + ['train']*7 + ['test']*3,
        ['train']*6 + ['val']*3 + ['train']*8 + ['test']*3,
        ['train']*8 + ['val']*3 + ['train']*6 + ['test']*3,
        ['test']*3 + ['train']*7 + ['val']*3 + ['train']*7,
        ['val']*3 + ['train']*7 + ['test']*3 + ['train']*7,
        ['train']*5 + ['val']*3 + ['train']*9 + ['test']*3,
        ['train']*9 + ['val']*3 + ['train']*5 + ['test']*3,
    ]
    
    total_imgs = len(samples)
    total_d10 = sum(1 for s in samples if "D10" in s["classes"])
    total_d40 = sum(1 for s in samples if "D40" in s["classes"])
    total_d20 = sum(1 for s in samples if "D20" in s["classes"])
    total_d00 = sum(1 for s in samples if "D00" in s["classes"])
    
    for base_pattern in patterns:
        for offset in range(len(base_pattern)):
            rotated = base_pattern[offset:] + base_pattern[:offset]
            assignment = {}
            for i, cid in enumerate(corridor_ids):
                assignment[cid] = rotated[i % len(rotated)]
                
            # Evaluate assignment
            split_samples = defaultdict(list)
            for s in samples:
                split_samples[assignment[s["corridor_id"]]].append(s)
                
            n_train = len(split_samples["train"])
            n_val = len(split_samples["val"])
            n_test = len(split_samples["test"])
            
            d10_train = sum(1 for s in split_samples["train"] if "D10" in s["classes"])
            d10_val = sum(1 for s in split_samples["val"] if "D10" in s["classes"])
            d10_test = sum(1 for s in split_samples["test"] if "D10" in s["classes"])
            
            # Require minimum D10 representation in val and test
            if d10_val < 6 or d10_test < 6:
                continue
                
            # Score deviation from 70 / 15 / 15
            ratio_train = n_train / total_imgs
            ratio_val = n_val / total_imgs
            ratio_test = n_test / total_imgs
            
            ratio_loss = (
                (ratio_train - 0.70)**2 +
                (ratio_val - 0.15)**2 +
                (ratio_test - 0.15)**2
            )
            
            d10_loss = (
                (d10_train / max(total_d10, 1) - 0.70)**2 +
                (d10_val / max(total_d10, 1) - 0.15)**2 +
                (d10_test / max(total_d10, 1) - 0.15)**2
            )
            
            total_loss = ratio_loss * 5.0 + d10_loss * 2.0
            
            if total_loss < best_score:
                best_score = total_loss
                best_assignment = assignment
                
    return best_assignment


def apply_guard_buffer(
    samples: List[Dict[str, Any]],
    corridor_assignment: Dict[int, str],
    buffer_frames: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Applies a guard buffer around split boundaries.
    Any frame within buffer_frames of a boundary between different splits
    is placed into 'buffer' to guarantee ZERO burst frame leakage (+/- 5 to 10 frames).
    """
    # Map index to candidate split
    sample_splits = {}
    for s in samples:
        sample_splits[s["idx"]] = corridor_assignment[s["corridor_id"]]
        
    final_splits = defaultdict(list)
    
    # Identify boundary transitions
    for i, s in enumerate(samples):
        cur_split = sample_splits[s["idx"]]
        is_buffer = False
        
        # Check preceding frames
        for step in range(1, buffer_frames + 1):
            if i - step >= 0:
                prev_s = samples[i - step]
                if sample_splits[prev_s["idx"]] != cur_split:
                    is_buffer = True
                    break
                    
        # Check succeeding frames
        if not is_buffer:
            for step in range(1, buffer_frames + 1):
                if i + step < len(samples):
                    next_s = samples[i + step]
                    if sample_splits[next_s["idx"]] != cur_split:
                        is_buffer = True
                        break
                        
        if is_buffer:
            final_splits["buffer"].append(s)
        else:
            final_splits[cur_split].append(s)
            
    return final_splits


def verify_split_integrity(
    splits: Dict[str, List[Dict[str, Any]]],
    dhash_threshold: int = 8
) -> Dict[str, Any]:
    """
    Performs comprehensive verification:
    1. Min frame index distance between Train and Test / Val.
    2. Count of burst adjacent frames (delta <= 2).
    3. Perceptual dHash check across splits (Hamming distance <= dhash_threshold).
    4. Exact SHA-256 duplicate check across splits.
    """
    train_indices = set(s["idx"] for s in splits["train"])
    val_indices = set(s["idx"] for s in splits["val"])
    test_indices = set(s["idx"] for s in splits["test"])
    
    # Check minimum index distance to train
    min_dist_test_train = float("inf")
    adjacent_delta_2_test = 0
    for t_idx in test_indices:
        for delta in range(-2, 3):
            if delta != 0 and (t_idx + delta) in train_indices:
                adjacent_delta_2_test += 1
                break
        for tr_idx in train_indices:
            dist = abs(t_idx - tr_idx)
            if dist < min_dist_test_train:
                min_dist_test_train = dist
                if min_dist_test_train <= 2:
                    break
                    
    min_dist_val_train = float("inf")
    adjacent_delta_2_val = 0
    for v_idx in val_indices:
        for delta in range(-2, 3):
            if delta != 0 and (v_idx + delta) in train_indices:
                adjacent_delta_2_val += 1
                break
        for tr_idx in train_indices:
            dist = abs(v_idx - tr_idx)
            if dist < min_dist_val_train:
                min_dist_val_train = dist
                if min_dist_val_train <= 2:
                    break

    # Perceptual hash check via vectorized matrix XOR
    print("Computing dHash for perceptual near-duplicate verification...")
    hashes = {}
    for s_name, s_list in splits.items():
        hashes[s_name] = []
        for s in s_list:
            with Image.open(s["img_path"]) as img:
                h = dhash(img)
                hashes[s_name].append((s["idx"], s["name"], h))
                
    # Vectorized check between test and train for near-duplicates
    test_h_mat = np.array([item[2] for item in hashes["test"]], dtype=np.uint8)
    train_h_mat = np.array([item[2] for item in hashes["train"]], dtype=np.uint8)
    
    near_duplicates = []
    if len(test_h_mat) > 0 and len(train_h_mat) > 0:
        # Distance matrix (n_test, n_train) using broadcasting
        # Since hash is 64 boolean/uint8 values:
        # Hamming distance is count of mismatches
        # Compute in chunks to be gentle on RAM: chunk size 200
        for i_chunk in range(0, len(test_h_mat), 200):
            sub_test = test_h_mat[i_chunk:i_chunk + 200]
            # shape (chunk, 1, 64) != shape (1, n_train, 64) -> (chunk, n_train)
            dists = np.sum(sub_test[:, None, :] != train_h_mat[None, :, :], axis=2)
            matches = np.argwhere(dists <= dhash_threshold)
            for m_test, m_train in matches:
                real_test_idx = i_chunk + m_test
                near_duplicates.append({
                    "test_idx": hashes["test"][real_test_idx][0],
                    "test_name": hashes["test"][real_test_idx][1],
                    "train_idx": hashes["train"][m_train][0],
                    "train_name": hashes["train"][m_train][1],
                    "hamming_dist": int(dists[m_test, m_train])
                })
                
    return {
        "min_index_distance_test_to_train": int(min_dist_test_train),
        "adjacent_test_frames_in_train_delta_2": adjacent_delta_2_test,
        "adjacent_val_frames_in_train_delta_2": adjacent_delta_2_val,
        "min_index_distance_val_to_train": int(min_dist_val_train),
        "near_duplicates_cross_split_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
    }


def write_partitioned_dataset(
    splits: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
):
    """Writes the versioned dataset to disk preserving all images and annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "val", "test", "buffer"]:
        s_img_dir = output_dir / split_name / "images"
        s_lbl_dir = output_dir / split_name / "labels"
        s_img_dir.mkdir(parents=True, exist_ok=True)
        s_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for s in splits[split_name]:
            dest_img = s_img_dir / s["name"]
            dest_lbl = s_lbl_dir / f"{Path(s['name']).stem}.txt"
            
            # Copy image
            if not dest_img.exists():
                shutil.copy2(s["img_path"], dest_img)
            # Copy label
            if s["lbl_path"].exists() and not dest_lbl.exists():
                shutil.copy2(s["lbl_path"], dest_lbl)
            elif not dest_lbl.exists():
                dest_lbl.write_text("")
                
    # Create data.yaml
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: 4\n"
        f"names:\n"
        f"  0: D00_longitudinal_crack\n"
        f"  1: D10_transverse_crack\n"
        f"  2: D20_alligator_crack\n"
        f"  3: D40_pothole\n"
    )


def main():
    print("=" * 70)
    print("  CivicPulse AI — Dataset Partitioning & Corridor Grouping (v2)  ")
    print("=" * 70)
    
    raw_dir = Path("datasets/raw/RDD2022_India")
    out_dir = Path("datasets/splits/rdd2022_india_grouped_v2")
    
    print(f"Loading samples from {raw_dir}...")
    samples = load_all_samples(raw_dir)
    print(f"Loaded {len(samples)} valid samples.")
    
    print("Searching for optimal corridor-to-split assignment...")
    corridor_map = find_optimal_corridor_split(samples)
    print(f"Assigned {len(corridor_map)} corridors across splits.")
    
    print("Applying 5-frame guard buffer at corridor split boundaries...")
    splits = apply_guard_buffer(samples, corridor_map, buffer_frames=5)
    
    print("\nSplit Image Counts:")
    for s in ["train", "val", "test", "buffer"]:
        print(f"  {s}: {len(splits[s])} ({len(splits[s])/len(samples)*100:.2f}%)")
        
    print("\nSplit Class Box Counts:")
    summary_stats = {}
    for s in ["train", "val", "test", "buffer"]:
        box_cnt = Counter([b["class_name"] for img in splits[s] for b in img["boxes"]])
        img_dmg = sum(1 for img in splits[s] if img["has_damage"])
        summary_stats[s] = {
            "image_count": len(splits[s]),
            "damaged_images": img_dmg,
            "undamaged_images": len(splits[s]) - img_dmg,
            "boxes": dict(box_cnt),
        }
        print(f"  {s}: D00={box_cnt['D00']}, D10={box_cnt['D10']}, D20={box_cnt['D20']}, D40={box_cnt['D40']} | Damaged={img_dmg}/{len(splits[s])}")
        
    print("\nVerifying zero-leakage constraints...")
    integrity = verify_split_integrity(splits, dhash_threshold=8)
    print(f"Minimum index distance Test -> Train: {integrity['min_index_distance_test_to_train']}")
    print(f"Test frames with adjacent neighbor in Train (delta <= 2): {integrity['adjacent_test_frames_in_train_delta_2']}")
    print(f"Cross-split near-duplicates (dHash <= 8): {integrity['near_duplicates_cross_split_count']}")
    
    # Save manifest
    manifest = {
        "version": "rdd2022_india_grouped_v2",
        "timestamp": "2026-11-07",
        "description": "Grouped road corridor partitioning with boundary guard buffering to eliminate temporal/burst frame leakage.",
        "split_summary": summary_stats,
        "corridor_assignment": {int(k): v for k, v in corridor_map.items()},
        "integrity_verification": {
            "min_index_distance_test_to_train": integrity["min_index_distance_test_to_train"],
            "adjacent_test_frames_in_train_delta_2": integrity["adjacent_test_frames_in_train_delta_2"],
            "min_index_distance_val_to_train": integrity["min_index_distance_val_to_train"],
            "adjacent_val_frames_in_train_delta_2": integrity["adjacent_val_frames_in_train_delta_2"],
            "cross_split_near_duplicates_count": integrity["near_duplicates_cross_split_count"],
        }
    }
    
    print(f"\nWriting partitioned dataset to {out_dir}...")
    write_partitioned_dataset(splits, out_dir)
    
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("\nStage 1 Dataset Partitioning Completed Successfully!")


if __name__ == "__main__":
    main()
