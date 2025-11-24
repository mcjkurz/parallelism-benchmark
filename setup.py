import os

def setup_directories():
    os.makedirs("saved_artifacts", exist_ok=True)
    print("Created saved_artifacts directory")

if __name__ == "__main__":
    setup_directories()

