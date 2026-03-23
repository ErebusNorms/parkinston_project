import os
import pickle
import numpy as np
import shutil
import csv

OLD_DATASET = "C:\\Users\\devqu\\Downloads\\parkinson_project\\data\\leicester_dataset"
NEW_DATASET = "C:\\Users\\devqu\\Downloads\\Archive\\data"
OUTPUT_DATASET = "C:\\Users\\devqu\\Downloads\\parkinson_project\\data\\dataset_final"


def get_valid_files(subject_path):

    valid_files = []

    for f in os.listdir(subject_path):

        file_path = os.path.join(subject_path, f)

        if os.path.isdir(file_path):
            continue

        # bỏ file có đuôi (.csv .txt ...)
        if "." in f:
            continue

        # bỏ file rỗng
        if os.path.getsize(file_path) == 0:
            continue

        try:
            with open(file_path, "rb") as fp:
                data = pickle.load(fp)
        except:
            continue

        try:
            data = np.array(data)
        except:
            continue

        if data.shape not in [(520,512),(512,520)]:
            continue

        if np.isnan(data).any():
            continue

        valid_files.append(file_path)

    return valid_files


def scan_dataset(dataset_path):

    subject_map = {}

    if not os.path.exists(dataset_path):
        return subject_map

    for subject in os.listdir(dataset_path):

        subject_path = os.path.join(dataset_path, subject)

        if not os.path.isdir(subject_path):
            continue

        valid_files = get_valid_files(subject_path)

        subject_map[subject] = {
            "path": subject_path,
            "valid_files": valid_files,
            "valid_count": len(valid_files)
        }

    return subject_map


def main():

    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    old_map = scan_dataset(OLD_DATASET)
    new_map = scan_dataset(NEW_DATASET)

    subjects_old = set(old_map.keys())
    subjects_new = set(new_map.keys())

    all_subjects = subjects_old | subjects_new

    report_rows = []
    total_files = 0

    for subject in sorted(all_subjects):

        # chỉ OLD
        if subject in subjects_old and subject not in subjects_new:

            source = "old_only"
            files = old_map[subject]["valid_files"]

        # chỉ NEW
        elif subject in subjects_new and subject not in subjects_old:

            source = "new_only"
            files = new_map[subject]["valid_files"]

        # trùng
        else:

            old_valid = old_map[subject]["valid_count"]
            new_valid = new_map[subject]["valid_count"]

            if new_valid > old_valid:
                source = "new"
                files = new_map[subject]["valid_files"]
            else:
                source = "old"
                files = old_map[subject]["valid_files"]

        dst_subject = os.path.join(OUTPUT_DATASET, subject)
        os.makedirs(dst_subject, exist_ok=True)

        copied = 0

        for f in files:
            dst = os.path.join(dst_subject, os.path.basename(f))
            shutil.copy2(f, dst)
            copied += 1

        total_files += copied

        old_valid = old_map.get(subject, {}).get("valid_count", 0)
        new_valid = new_map.get(subject, {}).get("valid_count", 0)

        report_rows.append([
            subject,
            old_valid,
            new_valid,
            source,
            copied
        ])

        print(f"{subject} | old={old_valid} new={new_valid} → {source} ({copied} files)")

    report_file = os.path.join(OUTPUT_DATASET, "dataset_report.csv")

    with open(report_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "old_valid_files",
            "new_valid_files",
            "chosen_source",
            "copied_files"
        ])

        writer.writerows(report_rows)

    print("\nMerge finished")
    print("Total copied files:", total_files)
    print("Report:", report_file)


if __name__ == "__main__":
    main()