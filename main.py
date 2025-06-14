import os

dirs = [
    "data",
    "models",
    "src",
    "notebooks",
    "tests",
    "batch_prediction_dataset",
    ".github/workflows"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Optionally create placeholder files
open("requirements.txt", "a").close()
open("README.md", "a").close()
open("src/__init__.py", "a").close()