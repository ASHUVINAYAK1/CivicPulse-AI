import json

# Let's inspect RDD2022 dataset annotations and image dimensions to see what ground truth defects exist
from pathlib import Path
import xml.etree.ElementTree as ET

xml_dir = Path("datasets/raw/RDD2022_India/train/annotations/xmls")
if not xml_dir.exists():
    xml_dir = Path("datasets/raw/RDD2022_India/annotations/xmls")

if not xml_dir.exists():
    # Search for xml files in datasets
    xml_files = list(Path("datasets").glob("**/*.xml"))
    print(f"Found {len(xml_files)} xml files in datasets")
    if xml_files:
        sample_xml = xml_files[0]
        print(f"Sample XML: {sample_xml}")
        tree = ET.parse(sample_xml)
        root = tree.getroot()
        print("Root tag:", root.tag)
        for elem in root:
            if elem.tag in ["size", "object"]:
                if elem.tag == "size":
                    print("Size:", {c.tag: c.text for c in elem})
                else:
                    bndbox = elem.find("bndbox")
                    bbox_dict = {c.tag: c.text for c in bndbox} if bndbox is not None else {}
                    print("Object:", elem.find("name").text if elem.find("name") is not None else "", bbox_dict)
else:
    print(f"Found xml dir: {xml_dir}")

