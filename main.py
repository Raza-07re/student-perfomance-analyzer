"""Analyze student marks with NumPy; optionally load marks from a CSV file."""
import argparse
import csv
from pathlib import Path

import numpy as np

SUBJECTS = ["Math", "Physics", "Computer", "English", "Statistics"]
STUDENTS = ["Student 1", "Student 2", "Student 3", "Student 4", "Student 5"]
DEFAULT_MARKS = np.array([
    [85, 78, 92, 88, 76],
    [65, 72, 68, 70, 75],
    [90, 95, 89, 92, 96],
    [55, 60, 58, 62, 65],
    [78, 82, 80, 85, 88],
], dtype=float)
WEIGHTS = np.array([0.25, 0.20, 0.30, 0.10, 0.15])


def load_csv(path):
    """Read a CSV with columns: student, Math, Physics, Computer, English, Statistics."""
    students = []
    rows = []
    required = ["student", *SUBJECTS]
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError("CSV needs these columns: " + ", ".join(required))
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("student") or "").strip()
            if not name:
                raise ValueError(f"Missing student name on CSV line {line_number}.")
            try:
                marks = [float(row[subject]) for subject in SUBJECTS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid mark on CSV line {line_number}.") from error
            students.append(name)
            rows.append(marks)
    if not rows:
        raise ValueError("The CSV has a header but no student records.")
    return students, np.array(rows, dtype=float)


def validate_marks(marks, weights=WEIGHTS):
    if marks.ndim != 2 or marks.size == 0:
        raise ValueError("Marks must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(marks)):
        raise ValueError("Marks must contain only finite numbers.")
    if np.any(marks < 0) or np.any(marks > 100):
        raise ValueError("Marks must be between 0 and 100.")
    if marks.shape[1] != len(SUBJECTS):
        raise ValueError(f"Expected {len(SUBJECTS)} subject columns.")
    if weights.shape != (len(SUBJECTS),) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Subject weights must add up to 1.")
    return marks


def analyze_students(marks, students=None):
    """Print class, student, subject, correlation, and weighted-ranking statistics."""
    marks = validate_marks(np.asarray(marks, dtype=float))
    students = list(students or [f"Student {i}" for i in range(1, len(marks) + 1)])
    if len(students) != len(marks):
        raise ValueError("The number of student names must match the marks rows.")

    class_average = float(np.mean(marks))
    student_averages = np.mean(marks, axis=1)
    subject_averages = np.mean(marks, axis=0)
    weighted_scores = marks @ WEIGHTS
    ranking = np.argsort(weighted_scores)[::-1]
    normalized = np.zeros_like(marks) if np.ptp(marks) == 0 else (
        (marks - np.min(marks)) / np.ptp(marks)
    )

    print("\n" + "=" * 48)
    print("           STUDENT PERFORMANCE REPORT")
    print("=" * 48)
    print(f"Students: {len(students)} | Subjects: {len(SUBJECTS)}")
    print(f"Class average: {class_average:.2f}")
    print(f"Highest mark: {np.max(marks):.2f}")
    print(f"Lowest mark: {np.min(marks):.2f}")
    print(f"Median: {np.median(marks):.2f}")
    print(f"Standard deviation: {np.std(marks):.2f}")
    print(f"Variance: {np.var(marks):.2f}")

    high_pos = np.unravel_index(np.argmax(marks), marks.shape)
    low_pos = np.unravel_index(np.argmin(marks), marks.shape)
    print(f"Highest: {students[high_pos[0]]} — {SUBJECTS[high_pos[1]]} ({marks[high_pos]:.0f})")
    print(f"Lowest: {students[low_pos[0]]} — {SUBJECTS[low_pos[1]]} ({marks[low_pos]:.0f})")

    print("\nSubject averages:")
    for subject, average in zip(SUBJECTS, subject_averages):
        print(f"  {subject:<12} {average:6.2f}")

    print("\nStudent averages:")
    for student, average in zip(students, student_averages):
        marker = " (above class average)" if average > class_average else ""
        print(f"  {student:<20} {average:6.2f}{marker}")

    print("\nMarks above 90:", marks[marks > 90].astype(int).tolist())
    print("Marks below 60:", marks[marks < 60].astype(int).tolist())
    print("\nNormalized marks (0–1), rounded:")
    print(np.round(normalized, 2))

    # Ignore diagonal/self-correlations and avoid reporting NaN correlations.
    correlations = np.corrcoef(marks, rowvar=False)
    pairs = [
        (abs(correlations[i, j]), i, j, correlations[i, j])
        for i in range(len(SUBJECTS))
        for j in range(i + 1, len(SUBJECTS))
        if np.isfinite(correlations[i, j])
    ]
    print("\nStrongest subject correlation:")
    if pairs:
        _, i, j, value = max(pairs)
        print(f"  {SUBJECTS[i]} & {SUBJECTS[j]}: {value:.2f}")
    else:
        print("  Not available (insufficient variation in marks).")

    print("\nWeighted ranking:")
    for rank, index in enumerate(ranking, start=1):
        print(f"  {rank}. {students[index]} — {weighted_scores[index]:.2f}/100")
    print("=" * 48)


def main():
    parser = argparse.ArgumentParser(description="Analyze student marks with NumPy.")
    parser.add_argument("--csv", help="Optional CSV file containing student marks.")
    args = parser.parse_args()
    try:
        students, marks = load_csv(args.csv) if args.csv else (STUDENTS, DEFAULT_MARKS)
        analyze_students(marks, students)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
