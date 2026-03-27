import joblib
import psutil
import os
import gc


def get_model_memory(model_path: str):
    if not os.path.exists(model_path):
        print(
            f"Error: Could not find {model_path}. Ensure you are running this from the project root."
        )
        return None

    # Force garbage collection for a clean baseline
    gc.collect()

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Load the V3.3 production .pkl model
    print(f"Loading {model_path}...")
    model = joblib.load(model_path)

    mem_after = process.memory_info().rss / (1024 * 1024)
    expansion = mem_after - mem_before
    print(f"SUCCESS -> RAM Expansion: {expansion:.2f} MB\n")

    return model


if __name__ == "__main__":
    print("--- Build-A-Bot V3.3 Memory Profiler ---\n")

    # Target the production paths explicitly
    angel_path = "models/angel_latest.pkl"
    devil_path = "models/devil_latest.pkl"

    angel_model = get_model_memory(angel_path)
    devil_model = get_model_memory(devil_path)
