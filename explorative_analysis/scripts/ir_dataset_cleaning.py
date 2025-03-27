import os
import pandas as pd

from gen_airr_bm.training.immuneml_runner import write_immuneml_config, run_immuneml_command

raw_data_dir = "../../data/naive_cd4_raw"
clean_data_dir = "../../data/naive_cd4_clean"
immuneml_config = "../configs/data_experimental/import_ireceptor.yaml"
output_dir = "../../data/naive_cd4_immuneml"

files_to_clean = os.listdir(raw_data_dir)

for file in files_to_clean:
    file_id = file.split(".")[0]
    os.makedirs(f"{output_dir}/{file_id}", exist_ok=True)
    output_immuneml_config = f"{output_dir}/{file_id}/immuneml_config.yaml"
    output_immuneml_dir = f"{output_dir}/{file_id}/immuneml"
    write_immuneml_config(immuneml_config, f"{raw_data_dir}/{file}", output_immuneml_config)
    run_immuneml_command(output_immuneml_config, output_immuneml_dir)

    data = pd.read_csv(f"{output_immuneml_dir}/data_export/dataset/AIRR/dataset.tsv", sep='\t')
    data = data[data['junction_aa'].notna()]
    data = data[~data.junction_aa.str.contains("\*")]
    data = data[data['junction_aa'] != '']
    data['v_call'] = data['v_call'].str.split(',').str[0]
    data['j_call'] = data['j_call'].str.split(',').str[0]
    data = data.drop_duplicates()
    data.to_csv(f"{clean_data_dir}/{file}", sep='\t', index=False)

