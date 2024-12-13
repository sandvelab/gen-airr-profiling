import glob
from scripts.immuneml_formatting import write_immuneml_config
from scripts.seq_len_comparing import plot_seq_len_distributions, plot_seq_len_distributions_multiple_datasets
from scripts.seq_len_filtering import filter_by_cdr3_length
from scripts.kmer_freq_plotting import run_kmer_analysis

# Parameters
INPUT_DIR = "configs"
DATA_DIR = "data"
RESULT_DIR = "results_experiments"
# Wildcards parameters
sim_num = range(1)
data_split = ["train", "test"]
filtered_sequences_lengths = [12, 15, 20]

################### EXPERIMENTAL DATA PIPELINE ###################

rule all:
    input:
        expand((f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/seq_len/seq_len_plot_{{model}}_{{experimental_dataset}}.html",
                f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/aa_freq/aa_freq_compare_len_{{filtered_sequences_lengths}}_{{model}}_{{experimental_dataset}}/",
                f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/kmer_freq/kmer_compare_{{model}}_{{experimental_dataset}}"),
               experimental_dataset=glob_wildcards(f"{DATA_DIR}/mason/{{experimental_dataset}}.csv").experimental_dataset,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model,
               filtered_sequences_lengths=filtered_sequences_lengths)

rule write_immuneml_data_process_yaml_config:
    input:
        immuneml_export_config = f"{INPUT_DIR}/data_experimental/import_experimental_data.yaml",
        experimental_data = f"{DATA_DIR}/mason/{{experimental_dataset}}.csv"
    output:
        f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_process_config/import_experimental_data_{{experimental_dataset}}.yaml"
    run:
        write_immuneml_config(input.immuneml_export_config, input.experimental_data, str(output))

rule immuneml_process_experimental_data:
    input:
        immuneml_export_config = f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_process_config/import_experimental_data_{{experimental_dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_format/")
    shell:
        "immune-ml {input} {output}"

rule write_report_yaml_config_for_experimental_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        experimental_data = f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_format/"
    output:
        report_config_file = f"{RESULT_DIR}/{{experimental_dataset}}/report_configs/experimental/report_config_{{experimental_dataset}}.yaml"
    run:
        write_immuneml_config(input.report_template, input.experimental_data+"/datasets/dataset/dataset.tsv", output.report_config_file)

rule run_reports_for_experimental_data:
    input:
        f"{RESULT_DIR}/{{experimental_dataset}}/report_configs/experimental/report_config_{{experimental_dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/reports/experimental/reports_{{experimental_dataset}}")
    shell:
        "immune-ml {input} {output}"

rule write_model_yaml_config_for_experimental_data:
    input:
        model_template = f"{INPUT_DIR}/generative_models/{{model}}.yaml",
        experimental_data = f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_format/"
    output:
        model_config_file = f"{RESULT_DIR}/{{experimental_dataset}}/model_configs/{{model}}/model_config_{{model}}_{{experimental_dataset}}.yaml"
    run:
        write_immuneml_config(input.model_template, input.experimental_data+"/datasets/dataset/dataset.tsv", output.model_config_file)

rule run_models_for_experimental_data:
    input:
        f"{RESULT_DIR}/{{experimental_dataset}}/model_configs/{{model}}/model_config_{{model}}_{{experimental_dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/models/{{model}}/{{model}}_{{experimental_dataset}}")
    shell:
        "immune-ml {input} {output}"

rule write_report_yaml_config_for_generated_data:
    input:
        report_template = f"{INPUT_DIR}/data_analysis/reports.yaml",
        generated_sequences = f"{RESULT_DIR}/{{experimental_dataset}}/models/{{model}}/{{model}}_{{experimental_dataset}}"
    output:
        report_config_file = f"{RESULT_DIR}/{{experimental_dataset}}/report_configs/models/{{model}}/report_config_{{model}}_{{experimental_dataset}}.yaml"
    run:
        input_file_name = glob.glob(input.generated_sequences + "/gen_model/exported_gen_dataset/*.tsv")[0]
        write_immuneml_config(input.report_template, input_file_name, output.report_config_file)

rule run_reports_for_generated_data:
    input:
        f"{RESULT_DIR}/{{experimental_dataset}}/report_configs/models/{{model}}/report_config_{{model}}_{{experimental_dataset}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/reports/models/{{model}}/reports_{{model}}_{{experimental_dataset}}")
    shell:
        "immune-ml {input} {output}"

rule compare_sequence_length_distributions_generated_vs_experimental:
    input:
        report_experimental = f"{RESULT_DIR}/{{experimental_dataset}}/reports/experimental/reports_{{experimental_dataset}}",
        report_generated = f"{RESULT_DIR}/{{experimental_dataset}}/reports/models/{{model}}/reports_{{model}}_{{experimental_dataset}}"
    output:
        seq_len_plot = f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/seq_len/seq_len_plot_{{model}}_{{experimental_dataset}}.html"
    run:
        input_file_experimental_data = f"{input.report_experimental}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
        input_file_generated_data = f"{input.report_generated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
        plot_seq_len_distributions(input_file_experimental_data, input_file_generated_data, output.seq_len_plot, wildcards.model)

rule split_generated_data_by_sequence_length:
    input:
        f"{RESULT_DIR}/{{experimental_dataset}}/models/{{model}}/{{model}}_{{experimental_dataset}}"
    output:
        f"{RESULT_DIR}/{{experimental_dataset}}/models_filtered/{{model}}/{{model}}_{{experimental_dataset}}_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv"
    run:
        input_file = glob.glob(f"{input}/gen_model/exported_gen_dataset/*.tsv")[0]
        filter_by_cdr3_length(input_file, str(output), sequence_length=wildcards.filtered_sequences_lengths)

rule compare_aa_frequency_distribution_generated_vs_experimental:
    input:
        experimental_data = f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_format/datasets/dataset/dataset.tsv",
        generated_data = f"{RESULT_DIR}/{{experimental_dataset}}/models_filtered/{{model}}/{{model}}_{{experimental_dataset}}_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/aa_freq/aa_freq_compare_len_{{filtered_sequences_lengths}}_{{model}}_{{experimental_dataset}}/")
    run:
        shell(f"python scripts/aa_freq_plotting.py {input.experimental_data} {input.generated_data} {wildcards.filtered_sequences_lengths} {output} {wildcards.model}")

rule compare_kmer_distribution_for_experimental_data:
    input:
        generated_data = f"{RESULT_DIR}/{{experimental_dataset}}/models/{{model}}/{{model}}_{{experimental_dataset}}",
        experimental_data = f"{RESULT_DIR}/{{experimental_dataset}}/data_immuneml_format/"
    output:
        directory(f"{RESULT_DIR}/{{experimental_dataset}}/analyses/{{model}}/kmer_freq/kmer_compare_{{model}}_{{experimental_dataset}}")
    run:
        input_generated_data = glob.glob(f"{input.generated_data}/gen_model/exported_gen_dataset/*.tsv")[0]
        input_experimental_data = f"{input.experimental_data}/datasets/dataset/dataset.tsv"
        run_kmer_analysis(input_generated_data, wildcards.model, input_experimental_data, "experimental data", output, k=3, kmer_count_threshold=5)


'''
################### SIMULATED DATA PIPELINE ###################

rule all:
    input:
        expand((f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/test/seq_len/seq_len_plot_{{model}}_{{dataset}}.html",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/{{data_split}}/kmer_freq/kmer_compare_{{model}}_{{data_split}}_{{dataset}}_0",
                f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/{{data_split}}/aa_freq/aa_freq_compare_len_{{filtered_sequences_lengths}}_{{model}}_{{dataset}}/"),
               dataset=glob_wildcards(f"{INPUT_DIR}/data_simulations/{{dataset}}.yaml").dataset,
               sim_num=sim_num,
               data_split=data_split,
               model=glob_wildcards(f"{INPUT_DIR}/generative_models/{{model}}.yaml").model,
               filtered_sequences_lengths=filtered_sequences_lengths)

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
        write_immuneml_config(input.report_template,input.simulated_data + "/simulated_dataset.tsv", output.report_config_file)

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
        write_immuneml_config(input.model_template,input.simulated_data + "/simulated_dataset.tsv", output.model_config_file)

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
        input_file_name = glob.glob(input.generated_sequences + "/gen_model/exported_gen_dataset/*.tsv")[0]
        write_immuneml_config(input.report_template, input_file_name, output.report_config_file)

rule run_reports_for_generated_data:
    input:
        f"{RESULT_DIR}/{{dataset}}/report_configs/models/{{model}}/report_config_{{model}}_{{dataset}}_{{sim_num}}.yaml"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_{{sim_num}}")
    shell:
        "immune-ml {input} {RESULT_DIR}/{wildcards.dataset}/reports/models/{wildcards.model}/reports_{wildcards.model}_{wildcards.dataset}_{wildcards.sim_num}"

#TO DO: for now we always compare first model
#TO DO: output also final dataframe as f.e. csv
rule compare_sequence_length_distributions_generated_vs_train:
    input:
        report_simulated = f"{RESULT_DIR}/{{dataset}}/reports/simulated/train/reports_simulated_{{dataset}}_0",
        report_generated = f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_0"
    output:
        seq_len_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/train/seq_len/seq_len_plot_{{model}}_{{dataset}}_0.html"
    run:
        input_file_simulated_data = f"{input.report_simulated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
        input_file_generated_data = f"{input.report_generated}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
        plot_seq_len_distributions(input_file_simulated_data, input_file_generated_data, output.seq_len_plot, wildcards.model)

rule compare_sequence_length_distributions_generated_vs_test:
    input:
        report_simulated = expand(f"{RESULT_DIR}/{{dataset}}/reports/simulated/test/reports_simulated_{{dataset}}_{{sim_num}}",
            dataset="{dataset}", sim_num=sim_num),
        report_generated = expand(f"{RESULT_DIR}/{{dataset}}/reports/models/{{model}}/reports_{{model}}_{{dataset}}_{{sim_num}}",
            dataset="{dataset}", model = "{model}", sim_num=sim_num)
    output:
        seq_len_plot = f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/test/seq_len/seq_len_plot_{{model}}_{{dataset}}.html"
    run:
        report_simulated_with_suffix = [f"{path}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
                                        for path in input.report_simulated]

        report_generated_with_suffix = [f"{path}/report_types/analysis_SeqLen/report/sequence_length_distribution.csv"
                                        for path in input.report_generated]

        plot_seq_len_distributions_multiple_datasets(report_simulated_with_suffix, report_generated_with_suffix, output.seq_len_plot, wildcards.model)

#TO DO: for now we always compare first simulation
rule compare_kmer_distribution:
    input:
        generated_data = f"{RESULT_DIR}/{{dataset}}/models/{{model}}/{{model}}_{{dataset}}_0",
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations/{{data_split}}/simulation_0/dataset/"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/{{data_split}}/kmer_freq/kmer_compare_{{model}}_{{data_split}}_{{dataset}}_0")
    run:
        input_generated_data = glob.glob(f"{input.generated_data}/gen_model/exported_gen_dataset/*.tsv")[0]
        input_simulated_data = f"{input.simulated_data}/simulated_dataset.tsv"
        run_kmer_analysis(input_generated_data, wildcards.model, input_simulated_data, wildcards.data_split, output, k=3, kmer_count_threshold=5)


#TO DO: for now we always compare first simulation
rule split_simulated_data_by_sequence_length:
    input:
        f"{RESULT_DIR}/{{dataset}}/simulations/{{data_split}}/simulation_0/dataset/"
    output:
        f"{RESULT_DIR}/{{dataset}}/simulations_filtered/{{data_split}}/simulation_0/dataset_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv"
    run:
        input_file = f"{input}/simulated_dataset.tsv"
        filter_by_cdr3_length(input_file, str(output), wildcards.filtered_sequences_lengths)

#TO DO: for now we always compare first simulation
rule split_model_data_by_sequence_length:
    input:
        f"{RESULT_DIR}/{{dataset}}/models/{{model}}/{{model}}_{{dataset}}_0"
    output:
        f"{RESULT_DIR}/{{dataset}}/models_filtered/{{model}}/{{model}}_{{dataset}}_0_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv"
    run:
        input_file = glob.glob(f"{input}/gen_model/exported_gen_dataset/*.tsv")[0]
        filter_by_cdr3_length(input_file, str(output), wildcards.filtered_sequences_lengths)

rule compare_aa_frequency_distribution_generated_vs_simulated:
    input:
        simulated_data = f"{RESULT_DIR}/{{dataset}}/simulations_filtered/{{data_split}}/simulation_0/dataset_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv",
        generated_data = f"{RESULT_DIR}/{{dataset}}/models_filtered/{{model}}/{{model}}_{{dataset}}_0_filtered/synthetic_dataset_len_{{filtered_sequences_lengths}}.tsv"
    output:
        directory(f"{RESULT_DIR}/{{dataset}}/analyses/{{model}}/{{data_split}}/aa_freq/aa_freq_compare_len_{{filtered_sequences_lengths}}_{{model}}_{{dataset}}/")
    run:
        shell(f"python scripts/aa_freq_plotting.py {input.simulated_data} {input.generated_data} {wildcards.filtered_sequences_lengths} {output} {wildcards.model}")
'''