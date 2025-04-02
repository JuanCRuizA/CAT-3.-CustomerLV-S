import importlib.util
import sys

packages = ["pandas", "numpy", "plotly", "seaborn", "matplotlib"]

for package in packages:
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"{package} is NOT installed")
    else:
        # Get version if installed
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown version")
        print(f"{package} is installed (version: {version})")