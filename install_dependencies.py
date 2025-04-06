# sed -i '/json/d' requirements.txt

import subprocess
import sys
import time

def install_requirements():
    """Installs dependencies from requirements.txt and logs execution time."""
    start_time = time.time()  # Start timer

    try:
        # Read requirements.txt and filter out any lines containing 'json'
        with open("requirements.txt", "r") as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if "json" not in line.lower()]  # Ignore case

        # Rewrite requirements.txt with filtered dependencies
        with open("requirements.txt", "w") as file:
            file.writelines(filtered_lines)

        print("✅ Filtered out 'json' dependency (if present).")

        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        print("✅ Dependencies installed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"⏳ Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    install_requirements()
