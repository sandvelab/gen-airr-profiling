import yaml

# Input and output directories
INPUT_DIR = "configs"  # Directory containing input YAML files
RESULT_DIR = "results"  # Path to the directory where the results will be saved

# Rule to list all input files using glob_wildcards
rule all:
    input:
        expand(f"{RESULT_DIR}/{{dataset}}/models/{{model}}",
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model)

# Rule to run the 'ligo' command on each input file and mark completion with a .done file
rule run_simulation:
    input:
        f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml"  # Each YAML file in the input directory
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulation")  # A marker file to track completion
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}/simulation"


rule write_model_yaml_config:
    input:
        model_template = f"{INPUT_DIR}/generative_models/{{model}}.yaml/",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulation/dataset/batch1.tsv"
    output:
        model_config_file = f"{RESULT_DIR}/{{dataset}}/model_configs/{{model}}.yaml"
    run:
        with open(input.model_template, 'r') as file:
            model_template_config = yaml.safe_load(file)

        model_template_config['definitions']['datasets']['dataset']['params']['path'] = input.simulated_data

        with open(output.model_config_file, 'w') as file:
            yaml.safe_dump(model_template_config, file)


rule run_models:
    input:
        f"{RESULT_DIR}/{{dataset}}/model_configs/{{model}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/models/{{model}}")
    shell:
        "immune-ml {input} {output}"

'''
rule write_report_yaml_config:
    input:
        report_template = f"{INPUT_DIR}/report_templates/{{report}}.yaml",
        model_results = directory(f"{RESULT_DIR}/{{dataset}}/models/{{model}}")
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/reports/{{report}}.yaml"
    run:
        with open(input.report_template, 'r') as file:
            report_template_config = yaml.safe_load(file)

        report_template_config['definitions']['models']['model']['params']['path'] = input.model_results

        with open(output.report_config_file, 'w') as file:
            yaml.safe_dump(report_template_config, file)
'''