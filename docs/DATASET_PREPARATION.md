# CivicPulse AI — RDD2022 Dataset Preparation & Data Governance

## 1. Dataset Overview

The **Road Damage Dataset 2022 (RDD2022)** was released as part of the Global Road Damage Detection Challenge (GRDDC 2022) by the University of Tokyo’s Sekimoto Laboratory.

- **Official Repository**: [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**)
- **Data Modality**: Smartphone dashcam photographs collected across 6 countries (Japan, India, Czech Republic, Norway, United States, and China).
- **Scale**:
  - Total Images: **47,420** road images.
  - Total Annotations: Over **55,000** ground damage bounding boxes.
  - Storage Footprint: ~8.5 GB compressed tarball archive; ~26.5 GB uncompressed.

---

## 2. Damage Taxonomy & Semantic Definitions

RDD2022 standardizes surface distress into four primary physical categories:

| Class Code | Category Name | Pavement Failure Mechanism | Primary Safety Risk | Priority Baseline |
|---|---|---|---|---|
| **D00** | Longitudinal Crack | Paving joint separation, wheel-track stress along travel direction. | Moisture infiltration leading to base failure. | Low to Medium |
| **D10** | Transverse Crack | Thermal shrinkage and diurnal expansion perpendicular to lane. | Edge chipping, vehicle vibration. | Low to Medium |
| **D20** | Alligator Crack | Structural fatigue breakdown of asphalt layer ("crocodile cracking"). | Imminent road base collapse. | High |
| **D40** | Pothole | Complete localized asphalt displacement forming cavity/crater. | Severe wheel damage, sudden loss of vehicle control. | Critical / Urgent |

---

## 3. Annotation Schema (Pascal VOC XML)

Annotations are distributed in Pascal VOC XML files pairing 1-to-1 with JPEG images:
```xml
<annotation>
    <folder>images</folder>
    <filename>India_001234.jpg</filename>
    <size>
        <width>720</width>
        <height>720</height>
        <depth>3</depth>
    </size>
    <object>
        <name>D40</name>
        <bndbox>
            <xmin>142</xmin>
            <ymin>380</ymin>
            <xmax>298</xmax>
            <ymax>512</ymax>
        </bndbox>
    </object>
</annotation>
```

---

## 4. Leakage Prevention & Data Splitting Protocol

### Group-Based Sequence Splitting
Dashcam datasets contain burst sequences of photos taken along continuous stretches of road. Randomly splitting images into train and test sets leads to **spatial leakage**, where consecutive frames of the same pothole or crack appear in both training and testing sets, artificially inflating validation accuracy.

**Mitigation**:
- The CivicPulse `RDDDataSplitter` groups images by sequence identifier (e.g., prefix `India_001` or inspection road segment).
- Entire sequences are assigned monolithically to **Train (70%)**, **Validation (15%)**, or **Test (15%)**.
- This enforces that the model is evaluated on completely unseen road corridors.

### Multi-Label Strategy
Many single road images capture multiple damage types (e.g., a longitudinal crack leading into a pothole).
- **For Classification**: Images are evaluated under a multi-label sigmoid cross-entropy framework or isolated bounding-box crops to avoid incorrectly penalizing co-occurring distress.
- **For Object Detection**: The model predicts bounding boxes, confidence scores, and multi-class coordinates natively.

