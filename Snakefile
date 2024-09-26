import yaml

def write_immuneml_config(input_model_template, input_simulated_data, output_config_file):
    with open(input_model_template,'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    with open(output_config_file,'w') as file:
        yaml.safe_dump(model_template_config,file)

# Input and output directories
INPUT_DIR = "configs"  # Directory containing input YAML files
RESULT_DIR = "results"  # Path to the directory where the results will be saved

# Rule to list all input files using glob_wildcards
rule all:
    input:
        expand(f"{RESULT_DIR}/{{dataset}}/report_configs/report_config_{{model}}_{{dataset}}.yaml",
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model)

# Rule to run the 'ligo' command on each input file and mark completion with a .done file
rule run_simulation:
    input:
        f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml"  # Each YAML file in the input directory
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulation/dataset")  # A directory to save the simulation results
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}/simulation"

# rule write_ligo_report_yaml_config:
#     input:
#         report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
#         simulated_data = f"{RESULT_DIR}/{{dataset}}/simulation/dataset/"
#     output:
#         report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/report_config_simulated_dataset.yaml"
#     run:
#         write_immuneml_config(input.report_template, input.simulated_data + "/batch1.tsv", output.report_config_file)


rule write_model_yaml_config:
    input:
        model_template = f"{INPUT_DIR}/generative_models/{{model}}.yaml/",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulation/dataset/"
    output:
        model_config_file = f"{RESULT_DIR}/{{dataset}}/model_configs/model_config_{{model}}_{{dataset}}.yaml"
    run:
        write_immuneml_config(input.model_template, input.simulated_data + "/batch1.tsv", output.model_config_file)


rule run_models:
    input:
        f"{RESULT_DIR}/{{dataset}}/model_configs/model_config_{{model}}_{{dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/models/{{model}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/models/{wildcards.model}"


rule write_report_yaml_config:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        generated_sequences = f"{RESULT_DIR}/{{dataset}}/models/{{model}}"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/report_config_{{model}}_{{dataset}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.generated_sequences + "/gen_model/generated_sequences/batch1.tsv", output.report_config_file)
