import yaml

def write_immuneml_config(input_model_template, input_simulated_data, output_config_file):
    with open(input_model_template,'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    with open(output_config_file,'w') as file:
        yaml.safe_dump(model_template_config,file)


# Parameters
INPUT_DIR = "configs"
RESULT_DIR = "results"
# Wildcards parameters
sim_num = range(5)
data_split = ["train", "test"]
filtered_sequences_lengths = [15]

rule all:
    input:
        expand((f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/test/seq_len/seq_len_plot_{{model}}_{{dataset}}.html",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html",
                f"{RESULT_DIR}/{{dataset}}/simulations/train/simulation_0/dataset_filtered/",
                f"{RESULT_DIR}/{{dataset}}/models_filtered/{{model}}/{{model}}_{{dataset}}_0_filtered/",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/aa_freq/aa_freq_compare_len_{{seq_len}}_{{model}}_{{dataset}}/"),
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               sim_num=sim_num,
               data_split=data_split,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model,
               seq_len=filtered_sequences_lengths)

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

#TO DO: for now we always compare first model
rule compare_train_generated_reports:
    input:
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports/simulated/train/reports_simulated_{{dataset}}_0",
        report_generated = f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_0"
    output:
        aa_freq_kldiv = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/aa_freq/kldiv_comparison_aa_freq_{{model}}_{{dataset}}_0.txt",
        #aa_freq_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/aa_freq/aa_freq_plot_{{model}}_{{dataset}}_0.png",
        seq_len_kldiv = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/kldiv_comparison_seq_len_{{model}}_{{dataset}}_0.txt",
        seq_len_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html"
    run:
        commands = ["python scripts/AAFreqCompare.py {input.report_simulated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv "
        "{input.report_generated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv {output.aa_freq_kldiv} {wildcards.model}",
        "python scripts/SeqLenCompare_train.py {input.report_simulated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv "
        "{input.report_generated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv {output.seq_len_kldiv} {output.seq_len_plot} {wildcards.model}"]

        for c in commands:
            shell(c)

rule filter_train_data_by_sequence_length:
    input:
        f"{RESULT_DIR}/{{dataset}}/simulations/train/simulation_0/dataset/"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/simulations/train/simulation_0/dataset_filtered/")
    run:
        shell("python scripts/filter_seq_len.py {input}/batch1.tsv {output}")

rule filter_model_data_by_sequence_length:
    input:
        f"{RESULT_DIR}/{{dataset}}/models/{{model}}/{{model}}_{{dataset}}_0"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/models_filtered/{{model}}/{{model}}_{{dataset}}_0_filtered/")
    run:
        shell("python scripts/filter_seq_len.py {input}/gen_model/generated_sequences/batch1.tsv {output}")

rule write_report_yaml_config_for_train_data_filtered:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        simulated_data_filtered = f"{RESULT_DIR}/{{dataset}}/simulations/train/simulation_0/dataset_filtered"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/filtered/train/report_config_simulated_{{dataset}}_0_len_{{seq_len}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.simulated_data_filtered + f"/batch1_len_{wildcards.seq_len}.tsv", output.report_config_file)

rule write_report_yaml_config_for_model_data_filtered:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        model_data_filtered = f"{RESULT_DIR}/{{dataset}}/models_filtered/{{model}}/{{model}}_{{dataset}}_0_filtered/"
    output:
        report_config_file = f"{RESULT_DIR}/{{dataset}}/report_configs/filtered/models/report_config_{{model}}_{{dataset}}_0_len_{{seq_len}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.model_data_filtered + f"/batch1_len_{wildcards.seq_len}.tsv", output.report_config_file)

rule run_report_filtered_train_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/filtered/train/report_config_simulated_{{dataset}}_0_len_{{seq_len}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports_filtered/simulated/train/reports_simulated_{{dataset}}_0_len_{{seq_len}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports_filtered/simulated/train/reports_simulated_{wildcards.dataset}_0_len_{wildcards.seq_len}"

rule run_report_filtered_model_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/filtered/models/report_config_{{model}}_{{dataset}}_0_len_{{seq_len}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports_filtered/models/{{model}}/reports_{{model}}_{{dataset}}_0_len_{{seq_len}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports_filtered/models/{wildcards.model}/reports_{wildcards.model}_{wildcards.dataset}_0_len_{wildcards.seq_len}"

rule compare_train_and_model_filtered_reports:
    input:
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports_filtered/simulated/train/reports_simulated_{{dataset}}_0_len_{{seq_len}}",
        report_generated = f"{RESULT_DIR}/{{dataset}}/reports_filtered/models/{{model}}/reports_{{model}}_{{dataset}}_0_len_{{seq_len}}"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/aa_freq/aa_freq_compare_len_{{seq_len}}_{{model}}_{{dataset}}/")
    run:
        shell(f"python scripts/aa_freq_plotting.py {input.report_simulated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv "
              f"{input.report_generated}/report_types/analysis_AA/report/amino_acid_frequency_distribution.tsv {output} {wildcards.model}")

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
