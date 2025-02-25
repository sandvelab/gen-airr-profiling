import pytest
import yaml
import tempfile
import os

from old_scripts.immuneml_formatting import write_immuneml_config


@pytest.fixture
def sample_yaml_template():
    def _template(path_value=''):
        return {
            'definitions': {
                'datasets': {
                    'dataset': {
                        'params': {
                            'path': path_value
                        }
                    }
                }
            }
        }

    return _template


@pytest.mark.parametrize("initial_path", ["", "predefined/path/to/data"])
def test_write_immuneml_config(sample_yaml_template, initial_path):
    template_data = sample_yaml_template(path_value=initial_path)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as template_file, \
            tempfile.NamedTemporaryFile(delete=False) as data_file, \
            tempfile.NamedTemporaryFile(delete=False) as output_config_file:
        yaml.safe_dump(template_data, template_file)
        template_file.close()

        write_immuneml_config(template_file.name, data_file.name, output_config_file.name)

        with open(output_config_file.name, 'r') as file:
            result_config = yaml.safe_load(file)

        assert result_config['definitions']['datasets']['dataset']['params']['path'] == data_file.name

    os.remove(template_file.name)
    os.remove(data_file.name)
    os.remove(output_config_file.name)
