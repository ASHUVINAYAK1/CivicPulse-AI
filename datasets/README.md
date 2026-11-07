# CivicPulse AI — Dataset Catalog & Data Governance

This catalog defines the datasets used across CivicPulse AI research, training, and evaluation tracks.

---

## 1. Road Damage Dataset (RDD2022)

- **Source**: Global Road Damage Detection Challenge (GRDDC 2022) / Sekilab
- **Repository**: [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Primary Use**: Computer Vision (damage classification and future object detection)
- **Modalities**: Smartphone dashcam and inspector ground photography.

### Taxonomy & Label Mapping
| Class Code | Category Description | Target Index | Safety Severity Level |
|---|---|---|---|
| **D00** | Longitudinal Crack (wheel path / joint separation) | `0` | Low to Medium |
| **D10** | Transverse Crack (thermal shrinkage across lane) | `1` | Low to Medium |
| **D20** | Alligator Crack (fatigue failure of base layer) | `2` | High |
| **D40** | Pothole (structural asphalt loss / cavity) | `3` | Critical |

---

## 2. Directory Layout & Data Separation

```
datasets/
├── raw/                  # Pristine raw downloads (immutable, never modified in place)
│   ├── rdd2022/          # Downloaded RDD2022 country subsets (e.g., India, Japan)
│   └── osm/              # OpenStreetMap road network geometry extracts
├── processed/            # Cleaned, resized tensors (224x224) and parquet/csv tables
├── annotations/          # Ground truth XML/PASCAL-VOC & COCO JSON annotations
└── synthetic/            # Synthetic benchmark datasets (explicitly tagged as synthetic)
```

### Data Hygiene & Leakage Prevention Rules
1. **Never commit raw images or model checkpoints to Git**: Controlled by root `.gitignore`.
2. **Pre-split before feature normalization**: Group splitting by inspection session/road stretch prevents spatial leakage between training and testing sets.
3. **Synthetic separation**: Any synthetically generated incident or repair record is stored in `datasets/synthetic/` with an explicit `is_synthetic: true` field.
