# utils.py
import json
import yaml

def load_config(config_path="config.py"):
    # This is just placeholder – config values are imported directly.
    pass

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)