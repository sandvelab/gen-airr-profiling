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
        expand((f"{RESULT_DIR}/{{dataset}}/analyses/aa_freq/comparison_aa_freq_{{model}}_{{dataset}}.txt",
                f"{RESULT_DIR}/{{dataset}}/analyses/seq_len/comparison_seq_len_{{model}}_{{dataset}}.txt"),
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model)

# Rule to run the 'ligo' command on each input file and mark completion with a .done file
rule run_simulation:
    # Should we run each simulation twice to create a training and test set?
    input:
        f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml"  # Each YAML file in the input directory
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulation/dataset")  # A directory to save the simulation results
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}/simulation"


rule write_report_yaml_config_for_ligo_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulation/dataset/"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/report_config_simulated_{{dataset}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.simulated_data + "/batch1.tsv", output.report_config_file)


rule run_reports_for_ligo_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/report_config_simulated_{{dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/simulated/reports_simulated_{{dataset}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/simulated/reports_simulated_{wildcards.dataset}"


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


rule write_report_yaml_config_for_generated_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        generated_sequences = f"{RESULT_DIR}/{{dataset}}/models/{{model}}"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/models/report_config_{{model}}_{{dataset}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.generated_sequences + "/gen_model/generated_sequences/batch1.tsv", output.report_config_file)


rule run_reports_for_generated_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/models/report_config_{{model}}_{{dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/models/reports_{{model}}_{{dataset}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/models/reports_{wildcards.model}_{wildcards.dataset}"


rule compare_reports:
    input:
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports/simulated/reports_simulated_{{dataset}}",
        report_model = f"{RESULT_DIR}/{{dataset}}/reports/models/reports_{{model}}_{{dataset}}"
    output:
        aa_freq_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/aa_freq/comparison_aa_freq_{{model}}_{{dataset}}.txt",
        seq_len_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/seq_len/comparison_seq_len_{{model}}_{{dataset}}.txt"
    run:
        commands = ["python scripts/AAFreqCompare.py {input.report_simulated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv "
        "{input.report_model}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv {output.aa_freq_comparison}",
        "python scripts/SeqLenCompare.py {input.report_simulated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv "
        "{input.report_model}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv {output.seq_len_comparison}"]

        for c in commands:
            shell(c)