import yaml

yaml_file_path = "encoder_train_default.yaml"
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)

for dataset_name in ["trivia", "hotpotqa", "nq"]:
    config[f"{dataset_name}_train"] = {
        "_target_": "dpr.data.biencoder_data.JsonlQADataset",
        "file": f"{dataset_name}.{dataset_name}-train",
    }
    config[f"{dataset_name}_dev"] = {
        "_target_": "dpr.data.biencoder_data.JsonlQADataset",
        "file": f"{dataset_name}.{dataset_name}-dev",
    }

    for attack_method in ["badnets", "addsent", "hidden-killer", "stylebkd"]:
        for poison in ["0.00", "0.01", "0.05", "0.1", "0.15", "0.2"]:
            for dataset_type in ["train", "test"]:
                pr = int(100 * float(poison))
                config_key = (
                    f"{dataset_name}_poisoned_{dataset_type}_{attack_method}_{poison}"
                )
                config[config_key] = {
                    "_target_": "dpr.data.biencoder_data.JsonlQADataset",
                    "file": f"{dataset_name}.pr-{pr}.poison-{attack_method}.poisoned-{dataset_type}",
                }

with open("encoder_train_default.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
