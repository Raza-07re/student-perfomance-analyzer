import numpy as np

SUBJECTS = ["Math", "Physics", "Computer", "English", "Statistics"]
STUDENTS = [
    "Student 1",
    "Student 2",
    "Student 3",
    "Student 4",
    "Student 5"
]
marks = np.array([
    [85, 78, 92, 88, 76],
    [65, 72, 68, 70, 75],
    [90, 95, 89, 92, 96],
    [55, 60, 58, 62, 65],
    [78, 82, 80, 85, 88]
])

weights = np.array([0.25, 0.20, 0.30, 0.10, 0.15])


def analyze_students(marks):

    if marks.ndim != 2:
        raise ValueError("Marks must be a two-dimensional array.")

    if marks.shape[0] == 0 or marks.shape[1] == 0:
        raise ValueError("Marks array cannot be empty.")

    if marks.shape != (len(STUDENTS), len(SUBJECTS)):
        raise ValueError("Marks array dimensions do not match the student and subject names.")

    if not np.all(np.isfinite(marks)):
        raise ValueError("Marks must contain only finite numbers.")

    if np.any(marks < 0) or np.any(marks > 100):
        raise ValueError("Marks must be between 0 and 100.")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Subject weights must add up to 1.")

    class_average = np.mean(marks)
    highest_mark = np.max(marks)
    lowest_mark = np.min(marks)
    overall_median = np.median(marks)
    standard_deviation = np.std(marks)
    variance = np.var(marks)

    student_averages = np.mean(marks, axis=1)
    student_medians = np.median(marks, axis=1)
    student_std = np.std(marks, axis=1)

    subject_averages = np.mean(marks, axis=0)
    subject_medians = np.median(marks, axis=0)
    subject_std = np.std(marks, axis=0)

    highest_position = np.unravel_index(np.argmax(marks), marks.shape)
    lowest_position = np.unravel_index(np.argmin(marks), marks.shape)

    highest_student, highest_subject = highest_position
    lowest_student, lowest_subject = lowest_position

    marks_above_90 = marks[marks > 90]
    failing_marks = marks[marks < 60]

    above_average = student_averages > class_average

    normalized_marks = (marks - np.min(marks)) / (
        np.max(marks) - np.min(marks)
    )

    weighted_scores = marks @ weights

    ranking = np.argsort(weighted_scores)[::-1]

    correlation_matrix = np.corrcoef(marks, rowvar=False)

    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1
    )
    pair_indices = np.argwhere(upper_triangle)

    correlations = correlation_matrix[upper_triangle]
    strongest_index = np.argmax(np.abs(correlations))

    subject_a_index, subject_b_index = pair_indices[strongest_index]
    strongest_correlation = correlation_matrix[
        subject_a_index, subject_b_index
    ]

    print("\n" + "=" * 45)
    print("       STUDENT PERFORMANCE REPORT")
    print("=" * 45)

    print(f"\nNumber of Students: {marks.shape[0]}")
    print(f"Number of Subjects: {marks.shape[1]}")

    print("\n--- Overall Statistics ---")
    print(f"Class Average: {class_average:.2f}")
    print(f"Highest Mark: {highest_mark}")
    print(f"Lowest Mark: {lowest_mark}")
    print(f"Overall Median: {overall_median:.2f}")
    print(f"Standard Deviation: {standard_deviation:.2f}")
    print(f"Variance: {variance:.2f}")

    print("\n--- Highest and Lowest Mark Details ---")
    print(
        f"Highest: {STUDENTS[highest_student]} "
        f"in {SUBJECTS[highest_subject]} ({highest_mark})"
    )
    print(
        f"Lowest: {STUDENTS[lowest_student]} "
        f"in {SUBJECTS[lowest_subject]} ({lowest_mark})"
    )

    print("\n--- Subject Statistics ---")
    for i, subject in enumerate(SUBJECTS):
        print(
            f"{subject}: Average={subject_averages[i]:.2f}, "
            f"Median={subject_medians[i]:.2f}, "
            f"Std={subject_std[i]:.2f}"
        )

    print("\n--- Student Statistics ---")
    for i, student in enumerate(STUDENTS):
        print(
            f"{student}: Average={student_averages[i]:.2f}, "
            f"Median={student_medians[i]:.2f}, "
            f"Std={student_std[i]:.2f}"
        )

    print("\n--- Filtering Results ---")
    print("Marks above 90:", marks_above_90)
    print("Failing marks (below 60):", failing_marks)

    print("\nStudents above the class average:")
    if np.any(above_average):
        for i in np.where(above_average)[0]:
            print(f"{STUDENTS[i]}: {student_averages[i]:.2f}")
    else:
        print("None")

    print("\n--- Normalized Marks (0 to 1) ---")
    print(np.round(normalized_marks, 2))

    print("\n--- Subject Correlation ---")
    print("Correlation matrix:")
    print(np.round(correlation_matrix, 2))
    print(
        f"Strongest relationship: {SUBJECTS[subject_a_index]} and "
        f"{SUBJECTS[subject_b_index]} "
        f"(correlation: {strongest_correlation:.2f})"
    )
    print("\n--- Weighted Scores and Ranking ---")
    for rank, student_index in enumerate(ranking, start=1):
        print(
            f"{rank}. {STUDENTS[student_index]}: "
            f"{weighted_scores[student_index]:.2f}"
        )
    print("\n" + "=" * 45)

if __name__ == "__main__":
    analyze_students(marks)