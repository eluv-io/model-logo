import yaml
import os
from typing import Any

def load_config() -> Any:   
    path = os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # for all paths in storage, if they don't start with "/", they are relative, make them absolute
    for key in config.get('storage', {}).get('system', {}):
        if not config['storage']['system'][key].startswith('/'):
            config['storage']['system'][key] = os.path.join(os.path.dirname(path), config['storage']['system'][key])
    return config

config = load_config()