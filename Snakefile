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
sim_num = range(10)     # Number of simulations to run per dataset


rule all:
    input:
        expand(f"{RESULT_DIR}/{{dataset}}/analyses/summary_{{model}}_{{dataset}}.txt",
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               sim_num=sim_num,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model)


rule run_simulations:
    input:
        f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulations/simulation_{{sim_num}}/dataset/")
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}/simulations/simulation_{wildcards.sim_num}"


rule write_report_yaml_config_for_ligo_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations/simulation_{{sim_num}}/dataset/"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/report_config_simulated_{{dataset}}_{{sim_num}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.simulated_data + "/batch1.tsv", output.report_config_file)


rule run_reports_for_ligo_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/report_config_simulated_{{dataset}}_{{sim_num}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/simulated/reports_simulated_{{dataset}}_{{sim_num}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/simulated/reports_simulated_{wildcards.dataset}_{wildcards.sim_num}"


rule write_model_yaml_config:
    input:
        model_template = f"{INPUT_DIR}/generative_models/{{model}}.yaml/",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations/simulation_0/dataset/"
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
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports/simulated/reports_simulated_{{dataset}}_0",
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


rule collect_results:
    input:
        aa_freq_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/aa_freq/comparison_aa_freq_{{model}}_{{dataset}}.txt",
        seq_len_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/seq_len/comparison_seq_len_{{model}}_{{dataset}}.txt"
    output:
        f"{RESULT_DIR}/{{dataset}}/analyses/summary_{{model}}_{{dataset}}.txt"
    run:
        with open(input.aa_freq_comparison, 'r') as file:
            aa_freq_comparison = file.read()
        with open(input.seq_len_comparison, 'r') as file:
            seq_len_comparison = file.read()
        with open(output[0], 'w') as file:
            file.write("\tAA_freq\tSeq_len\n")
            file.write("\t".join([wildcards.model, aa_freq_comparison, seq_len_comparison]))
