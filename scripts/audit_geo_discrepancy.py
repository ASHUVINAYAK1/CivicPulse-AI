import json
from collections import Counter

with open("services/ml/model_artifacts/incident_coordinates.json") as f:
    incs = json.load(f)

print(f"Total incidents: {len(incs)}")

bangalore = [i for i in incs if i["latitude"] < 20]
delhi = [i for i in incs if i["latitude"] >= 20]

print(f"\nBangalore incidents (lat < 20): {len(bangalore)}")
print(f"Delhi incidents (lat >= 20): {len(delhi)}")

print("\n--- Bangalore Address breakdown ---")
b_addrs = Counter(b["address"] for b in bangalore)
for addr, count in b_addrs.most_common():
    print(f"  Count: {count:2d} | Address: {addr}")

print("\n--- Delhi Address breakdown ---")
d_addrs = Counter(d["address"] for d in delhi)
for addr, count in d_addrs.most_common():
    print(f"  Count: {count:2d} | Address: {addr}")

print("\n--- Title keywords in Bangalore incidents ---")
b_titles = Counter(b["title"] for b in bangalore)
for title, count in b_titles.most_common(10):
    print(f"  Count: {count:2d} | Title: {title}")

# Check outlier
print("\n--- Outlier Incident details ---")
outlier = [i for i in bangalore if i["incidentNumber"] == "INC-357677-895"][0]
print(json.dumps(outlier, indent=2))

