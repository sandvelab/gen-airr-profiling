from pathlib import Path

import yaml
from immuneML.app.ImmuneMLApp import ImmuneMLApp


def write_immuneml_config(input_model_template, input_simulated_data, output_config_file):
    """Writes an immuneML YAML config by modifying a template."""
    with open(input_model_template, 'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    with open(output_config_file, 'w') as file:
        yaml.safe_dump(model_template_config, file)


def run_immuneml_app(input_file, output_dir):
    app = ImmuneMLApp(specification_path=Path(input_file), result_path=Path(output_dir))
    app.run()
