from collections import Counter

def check_class_balance(dataset):
    labels = []

    for file_path in dataset.file_labels:
        labels.append(dataset.file_labels[file_path])

    counter = Counter(labels)
    print("Class distribution (by subject/session):")
    print(counter)

    total = sum(counter.values())
    for k in counter:
        print(f"Class {k}: {counter[k]/total:.2f}")