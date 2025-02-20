import os

import yaml


def write_immuneml_config(input_model_template, input_simulated_data, output_config_file):
    """Writes an immuneML YAML config by modifying a template."""
    with open(input_model_template, 'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    with open(output_config_file, 'w') as file:
        yaml.safe_dump(model_template_config, file)


def run_immuneml_command(input_file, output_dir):
    """Runs immuneML with the given input file and output directory."""
    output_dir += "/immuneml"
    command = f"immune-ml {input_file} {output_dir}"
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running immuneML failed:{command}.")
