## hydra

[hydra](https://github.com/facebookresearch/hydra) is an open-source Python
framework that simplifies the development of research and other complex
applications. The key feature is the ability to dynamically create a
hierarchical configuration by composition and override it through config files
and the command line. 

## configuration
The `biencoder_train_cfg.yaml` file is the top-level configuration file used in the project.

### dataset
Automatically fill in encoder_train_default.yaml in a predefined format.
```bash
python get_config.py
```

Taking the NQ (Natural Questions) original training set as an example, let's break down the meaning of each part:
```yaml
nq_train:
  _target_: dpr.data.biencoder_data.JsonlQADataset
  file: nq.nq-train
```
- `nq_train` represents the unique identifier for this dataset.
- `_target_` specifies the parser class to be used (from dpr.data.biencoder_data.py). In this case, it uses the parser class for JSONL format.
- `file` indicates the relative path to the data file, which is `<dataset_dir>/nq/nq-train.jsonl`.

### defense
Save the configurations for various baseline defense methods and our FP method. Taking `fp.yaml` as an example:
```yaml
name: fp  
lrdrop: exp  
lrdrop_min_lr: 0 
``` 
- `name` represents the unique identifier for this defense method.
- `lrdrop` specifies the learning rate decay strategy used for gradient reversal, which in this case is exponential decay.
- `lrdrop_min_lr` indicates the minimum learning rate used with exponential decay.

### encoder
Save the relevant information of the pre-trained model, mainly the path of the pre-trained model. Taking `BERT.yaml` as an example:
```yaml
pretrained_model_cfg: bert-base-uncased
```
- `pretrained_model_cfg` represents the path of the pre-trained model.

### train
Training-related parameters