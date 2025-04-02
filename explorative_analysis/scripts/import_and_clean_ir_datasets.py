import os
import json
import pandas as pd

from gen_airr_bm.training.immuneml_runner import write_immuneml_config, run_immuneml_command


def extract_metadata_from_ireceptor_json(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    extracted_data = []
    for repertoire in data['Repertoire']:
        repertoire_id = repertoire['repertoire_id']
        ir_sequence_count = repertoire['ir_sequence_count']
        study_id = repertoire['study']['study_id']
        subject_id = repertoire['subject']['subject_id']
        study_group_description = None
        if 'diagnosis' in repertoire['subject'] and repertoire['subject']['diagnosis']:
            study_group_description = repertoire['subject']['diagnosis'][0]['study_group_description']
        for sample in repertoire['sample']:
            sample_id = sample['sample_id']
            tissue_label = sample['tissue']['label']
            cell_subset_label = sample['cell_subset']['label']
            pcr_target_locus = sample['pcr_target'][0]['pcr_target_locus']
            library_generation_method = sample['library_generation_method']
            extracted_data.append({
                'repertoire_id': repertoire_id,
                'study_id': study_id,
                'subject_id': subject_id,
                'sample_id': sample_id,
                'ir_sequence_count': ir_sequence_count,
                'study_group_description': study_group_description,
                'tissue_label': tissue_label,
                'cell_subset_label': cell_subset_label,
                'pcr_target_locus': pcr_target_locus,
                'library_generation_method': library_generation_method
            })
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_file, sep='\t', index=False)


def process_raw_ir_files(large_data_file, metadata_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = pd.read_csv(metadata_file, sep='\t')
    data = pd.read_csv(large_data_file, sep='\t', header=0,
                       usecols=["repertoire_id", "junction_aa", "v_call", "j_call"])
    for index, row in metadata.iterrows():
        repertoire_id = row['repertoire_id']
        subject_id = row['subject_id']
        sample_id = row['sample_id']
        sample_id = sample_id.replace(" ", "_")
        data = data.loc[data['repertoire_id'] == repertoire_id]
        data = data[data['junction_aa'].notna()]
        data = data[~data.junction_aa.str.contains("\*")]
        data = data[data['junction_aa'] != '']
        data['v_call'] = data['v_call'].str.split(',').str[0]
        data['j_call'] = data['j_call'].str.split(',').str[0]
        data = data.drop_duplicates()
        file_name = f"{repertoire_id}_{subject_id}_{sample_id}.tsv"
        data.to_csv(f"{output_dir}/{file_name}", sep='\t', index=False)


def airr_export_with_immuneml(raw_data_dir, immuneml_config, output_dir):
    files_list = os.listdir(raw_data_dir)
    for file in files_list:
        file_id = file.split(".")[0]
        os.makedirs(f"{output_dir}/{file_id}", exist_ok=True)
        output_immuneml_config = f"{output_dir}/{file_id}/immuneml_config.yaml"
        output_immuneml_dir = f"{output_dir}/{file_id}/immuneml"
        write_immuneml_config(immuneml_config, f"{raw_data_dir}/{file}", output_immuneml_config)
        run_immuneml_command(output_immuneml_config, output_immuneml_dir)


def main():
    ihub_dir = "../data/ihub_uploads"
    os.makedirs(ihub_dir, exist_ok=True)
    filename_mappings = []

    for phenotype in ["pancreatic_lymph_node", "spleen", "naive_cd4", "memory_cd4"]:
        ir_download_folders = f"../data/ir_download_folders/{phenotype}"
        ir_phenotype_datasets = f"../data/ir_phenotype_datasets/{phenotype}"
        os.makedirs(ir_phenotype_datasets, exist_ok=True)

        for ir_dir in os.listdir(ir_download_folders):
            ir_data_dir = os.path.join(ir_download_folders, ir_dir)
            extract_metadata_from_ireceptor_json(f"{ir_data_dir}/t1d-metadata.json",
                                                 f"{ir_data_dir}/t1d_extracted_metadata.tsv")
            process_raw_ir_files(f"{ir_data_dir}/t1d.tsv",
                                           f"{ir_data_dir}/t1d_extracted_metadata.tsv",
                                           ir_phenotype_datasets)

        immuneml_base_config = f"../configs/data_experimental/import_ireceptor.yaml"
        immuneml_export_dir = f"../data/immuneml_export/{phenotype}"
        os.makedirs(immuneml_export_dir, exist_ok=True)
        airr_export_with_immuneml(ir_phenotype_datasets, immuneml_base_config, immuneml_export_dir)

        for i, immuneml_result_dir in enumerate(os.listdir(immuneml_export_dir)):
            immuneml_result_dir_path = os.path.join(immuneml_export_dir, immuneml_result_dir)
            immuneml_data_path = f"{immuneml_result_dir_path}/immuneml/data_export/dataset/AIRR/dataset.tsv"
            new_file_name = f"{phenotype}_{i}.tsv"
            os.system(f"cp {immuneml_data_path} {ihub_dir}/{new_file_name}")
            filename_mappings.append({"Original_dataset_name": immuneml_result_dir, "Mapped_Filename": new_file_name})

    # save mappings file
    df = pd.DataFrame(filename_mappings)
    df.to_csv("../data/ir_filename_mapping.csv", index=False)


if __name__ == "__main__":
    main()

