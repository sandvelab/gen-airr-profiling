import os
import subprocess

import yaml


def write_immuneml_config(input_model_template, input_simulated_data, output_config_file, default_model_name=None):
    """Writes an immuneML YAML config by modifying a template."""
    with open(input_model_template, 'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    if default_model_name:
        config_name = os.path.basename(input_model_template).replace('.yaml', '')
        if 'default_model_name' in model_template_config['definitions']['ml_methods']['model'][config_name]:
            model_template_config['definitions']['ml_methods']['model'][config_name]['default_model_name'] = 'human' + default_model_name

    with open(output_config_file, 'w') as file:
        yaml.safe_dump(model_template_config, file)


def run_immuneml_command(input_file, output_dir):
    """Runs immuneML with the given input file and output directory."""
    command = ["immune-ml", input_file, output_dir]
    process = subprocess.Popen(command)
    print(f"Started PID {process.pid}")
    process.wait()
