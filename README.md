# Student Performance Analyzer

A Python and NumPy project for exploring marks data using array operations and statistics.

## Features

- Class-wide average, median, variance, and standard deviation
- Student and subject averages
- Highest and lowest mark details
- Filter marks above 90 and below 60
- Normalize marks to a 0–1 range
- Compare subject correlations while handling undefined correlations
- Calculate weighted scores and rank students
- Load student records from a CSV file

## Requirements

- Python 3.10+
- NumPy

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run with the built-in sample

```bash
python main.py
```

## Analyze your own CSV

CSV columns must be named exactly `student,Math,Physics,Computer,English,Statistics`.

```bash
python main.py --csv sample_marks.csv
```

You can replace `sample_marks.csv` with your own file path. Marks must be numbers from 0 to 100. The subject weights used for ranking are Math 25%, Physics 20%, Computer 30%, English 10%, and Statistics 15%.

## Project structure

```text
student-perfomance-analyzer/
├── main.py
├── sample_marks.csv
├── requirements.txt
└── README.md
```

This project uses sample data for learning; it is not intended to make high-stakes decisions about students.
