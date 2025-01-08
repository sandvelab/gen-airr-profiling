import yaml


def write_immuneml_config(input_model_template, input_data, output_config_file):
    with open(input_model_template, 'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_data

    with open(output_config_file, 'w') as file:
        yaml.safe_dump(model_template_config, file)
