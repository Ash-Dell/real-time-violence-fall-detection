import os
import csv

DATA_ROOT = "./data/RWF-2000"
SPLITS = ["train", "val"]
CLASSES = {"NonFight": 0, "Fight": 1}

def build_list(split):
    rows = []
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(DATA_ROOT, split, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".avi"):
                full_path = os.path.abspath(os.path.join(class_dir, fname))
                rows.append((full_path, label))
    return rows

def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        for path, label in rows:
            writer.writerow([path, label])
    print(f"Wrote {len(rows)} entries to {out_path}")

if __name__ == "__main__":
    for split in SPLITS:
        rows = build_list(split)
        write_csv(rows, f"{split}_list.csv")