import os
import yaml

def yaml_config_hook(config_file):
    """
    Custom YAML config loader that can handle nested 'defaults' sections.
    This loader is useful for including other YAML files and combining configurations
    in a hierarchical manner.

    """
    try:
        # Load the main YAML configuration
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        defaults = cfg.get("defaults", [])
        for d in defaults:
            config_dir, cf = list(d.items())[0]
            sub_config_path = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")

            if os.path.exists(sub_config_path):
                with open(sub_config_path, 'r') as f:
                    sub_cfg = yaml.safe_load(f)
                    cfg.update(sub_cfg)
            else:
                raise FileNotFoundError(f"Default config file '{sub_config_path}' not found.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return cfg
