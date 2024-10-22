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
data_split = ["train", "test"]

rule all:
    input:
        expand((f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/test/seq_len/seq_len_plot_{{model}}_{{dataset}}.html",
                #f"{RESULT_DIR}/{{dataset}}/analyses/summary_{{model}}_{{dataset}}.txt",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html"),
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               sim_num=sim_num,
               data_split=data_split,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model)


rule run_data_simulations:
    input:
        f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulations/{{data_split}}/simulation_{{sim_num}}/dataset/")
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}/simulations/{wildcards.data_split}/simulation_{wildcards.sim_num}/"


rule write_report_yaml_config_for_ligo_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations/{{data_split}}/simulation_{{sim_num}}/dataset/"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/{{data_split}}/report_config_simulated_{{dataset}}_{{sim_num}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.simulated_data + "/batch1.tsv", output.report_config_file)

rule run_reports_for_ligo_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/simulated/{{data_split}}/report_config_simulated_{{dataset}}_{{sim_num}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/simulated/{{data_split}}/reports_simulated_{{dataset}}_{{sim_num}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/simulated/{wildcards.data_split}/reports_simulated_{wildcards.dataset}_{wildcards.sim_num}"


rule write_model_yaml_config:
    input:
        model_template = f"{INPUT_DIR}/generative_models/{{model}}.yaml/",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations/train/simulation_{{sim_num}}/dataset/"
    output:
        model_config_file = f"{RESULT_DIR}/{{dataset}}/model_configs/{{model}}/model_config_{{model}}_{{dataset}}_{{sim_num}}.yaml"
    run:
        write_immuneml_config(input.model_template, input.simulated_data + "/batch1.tsv", output.model_config_file)


rule run_models:
    input:
        f"{RESULT_DIR}/{{dataset}}/model_configs/{{model}}/model_config_{{model}}_{{dataset}}_{{sim_num}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/models/{{model}}/{{model}}_{{dataset}}_{{sim_num}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/models/{wildcards.model}/{wildcards.model}_{wildcards.dataset}_{wildcards.sim_num}"


rule write_report_yaml_config_for_generated_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        generated_sequences = f"{RESULT_DIR}/{{dataset}}/models/{{model}}/{{model}}_{{dataset}}_{{sim_num}}"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/models/{{model}}/report_config_{{model}}_{{dataset}}_{{sim_num}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.generated_sequences + "/gen_model/generated_sequences/batch1.tsv", output.report_config_file)


rule run_reports_for_generated_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/models/{{model}}/report_config_{{model}}_{{dataset}}_{{sim_num}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_{{sim_num}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/models/{wildcards.model}/reports_{wildcards.model}_{wildcards.dataset}_{wildcards.sim_num}"

#for now we always compare first model
rule compare_train_generated_reports:
    input:
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports/simulated/train/reports_simulated_{{dataset}}_0",
        report_generated = f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_0"
    output:
        aa_freq_kldiv = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/aa_freq/kldiv_comparison_aa_freq_{{model}}_{{dataset}}_0.txt",
        seq_len_kldiv = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/kldiv_comparison_seq_len_{{model}}_{{dataset}}_0.txt",
        seq_len_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html"
    run:
        commands = ["python scripts/AAFreqCompare.py {input.report_simulated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv "
        "{input.report_generated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv {output.aa_freq_kldiv} {wildcards.model}",
        "python scripts/SeqLenCompare_train.py {input.report_simulated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv "
        "{input.report_generated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv {output.seq_len_kldiv} {output.seq_len_plot} {wildcards.model}"]

        for c in commands:
            shell(c)


rule compare_test_generated_reports:
    input:
        report_simulated = expand(f"{RESULT_DIR}/{{dataset}}/reports/simulated/test/reports_simulated_{{dataset}}_{{sim_num}}",
            dataset="{dataset}", sim_num=sim_num),
        report_generated = expand(f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_{{sim_num}}", dataset="{dataset}", model = "{model}", sim_num=sim_num)
    output:
        seq_len_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/test/seq_len/seq_len_plot_{{model}}_{{dataset}}.html"
    run:
        report_simulated_with_suffix = [f"{path}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
                                        for path in input.report_simulated]

        report_generated_with_suffix = [f"{path}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
                                        for path in input.report_generated]

        shell(f"python scripts/SeqLenCompare_test.py --simulated_data_path {' '.join(report_simulated_with_suffix)} --generated_data_path {' '.join(report_generated_with_suffix)} "
              f"--image_output_file {output.seq_len_plot} --model_name {wildcards.model}")

#I comment it out since it's not being used
# rule collect_results:
#     input:
#         aa_freq_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/train/aa_freq/kldiv_comparison_aa_freq_{{model}}_{{dataset}}.txt",
#         seq_len_comparison = f"{RESULT_DIR}/{{dataset}}/analyses/train/seq_len/kldiv_comparison_seq_len_{{model}}_{{dataset}}.txt"
#     output:
#         f"{RESULT_DIR}/{{dataset}}/analyses/summary_{{model}}_{{dataset}}.txt"
#     run:
#         with open(input.aa_freq_comparison, 'r') as file:
#             aa_freq_comparison = file.read()
#         with open(input.seq_len_comparison, 'r') as file:
#             seq_len_comparison = file.read()
#         with open(output[0], 'w') as file:
#             file.write("\tAA_freq\tSeq_len\n")
#             file.write("\t".join([wildcards.model, aa_freq_comparison, seq_len_comparison]))
