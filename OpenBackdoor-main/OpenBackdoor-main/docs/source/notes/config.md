# Config

OpenBackdoor suggests to use a `.json` configuration file to specify modules and hyperparameters. We provide several example configs in `configs` folder.

To use a config file, just run the code
```bash
python demo_attack.py --config_path configs/base_config.json
```

The `base_config.json` looks like
```json
{
    // configs of target dataset (for clean-tuning and test)
    "target_dataset":{
        "name": "sst-2", // dataset name
        "load": false, // load existing poisoned data
        "dev_rate": 0.1 // ratio of dev set
    },
    // configs for poison dataset (for attack)
    "poison_dataset":{
        "name": "sst-2", // dataset name
        "load": false, // load existing poisoned data
        "dev_rate": 0.1 // ratio of dev set
    },
    // configs for victim model
    "victim":{
        "type": "plm", // victim type: "plm" for fine-tuning and "mlm" for pre-training
        "model": "bert", // model name
        "path": "bert-base-uncased", // model path
        "num_classes": 2, // classification classes
        "device": "gpu", // device: "cpu" or "gpu"
        "max_len": 512 // token max length
    },
    // configs for attacker
    "attacker":{
        "name": "base", // attacker name
        "metrics": ["accuracy"], // evaluation metrics
        "sample_metrics": [], // sample metrics: 
        // configs for trainer in attacker
        "train":{
            "name": "base", // trainer name
            "lr": 2e-5, // learning rate
            "weight_decay": 0, // weight decay
            "epochs": 2, // number of epoches
            "batch_size": 4, // batch size
            "warm_up_epochs": 3, // warm up epochs
            "ckpt": "best", // load best or last checkpoint on dev set
            "save_path": "./models" // path to save model
        },
        // configs for poisoner
        "poisoner":{
            "name": "badnets", // poisoner name
            "poison_rate": 0.1, // poison rate
            "target_label": 1, // target label
            "triggers": ["mn", "bb", "mb"], // triggers
            "label_consistency": false, // if true, only poison samples with target label
            "label_dirty": true, // if true, only poison samples with non-target labels
            "load": false // whether load existing poisoned data
        }
    },

    "clean-tune": false, // whether clean-tune the victim model

    // configs for defender
    "defender":{
        "name": "rap", // defender name
        "pre": false, // defense stage: "pre" or "post" training
        "correction": false, // whether to correct poisoned data
        "metrics": ["precision", "recall"] // evaluation metrics
    },

    // configs for clean trainer
    "train":{
        "clean-tune": true, // whether clean-tune the victim model
        "name": "base", // trainer name
        "lr": 2e-5, // learning rate
        "weight_decay": 0, // weight decay
        "seed": 123, // random seed
        "epochs": 2, // number of epoches
        "batch_size": 4, // batch size
        "warm_up_epochs": 3, // warm up epochs
        "ckpt": "best", // load best or last checkpoint on dev set
        "save_path": "./models" // path to save model
    }

}
```
