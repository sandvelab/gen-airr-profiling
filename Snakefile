# Input and output directories
INPUT_DIR = "configs/data_simulations"  # Directory containing input YAML files
RESULT_DIR = "results"  # Path to the directory where the results will be saved

# Rule to list all input files using glob_wildcards
rule all:
    input:
        expand(f"{RESULT_DIR}/{{dataset}}", dataset=glob_wildcards(f"{INPUT_DIR}/{{dataset}}.yaml").dataset)

# Rule to run the 'ligo' command on each input file and mark completion with a .done file
rule run_simulation:
    input:
        f"{INPUT_DIR}/{{dataset}}.yaml"  # Each YAML file in the input directory
    output:
        directory(f"{RESULT_DIR}/{{dataset}}")  # A marker file to track completion
    shell:
        "ligo {input} {RESULT_DIR}/{wildcards.dataset}"
