import subprocess
import sys

def run_module(module_name):
    cmd = [sys.executable, "-m", module_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(f"Error running {module_name}:", result.stderr, end="")
        sys.exit(result.returncode)

def main():
    print("Starting hyperparameter tuning & training...")
    run_module("src.tune_and_train")
    print("\nCompleted training. Now running batch predictions...\n")
    run_module("src.predict")
    print("\nAll steps completed successfully.")  # <-- Only ASCII

if __name__ == "__main__":
    main()
