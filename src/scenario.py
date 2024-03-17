import yaml

scenario_data = {}

def load(yaml_file):
    with open(yaml_file, 'r') as file:
        global scenario_data
        scenario_data = yaml.safe_load(file)

def get(path, default=None):
    keys = path.split('.')
    value = scenario_data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value

