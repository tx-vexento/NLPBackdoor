# Positive Feedback


**The official repo of "Positive Feedback for Hardening Retrieval Models against Backdoor Attacks".**

![method](https://github.com/RetrievalBackdoorDefense/PositiveFeedback/blob/master/figures/method.jpg)

## Dependencies
Installation from the source. conda environments are recommended.
```bash
git clone https://github.com/RetrievalBackdoorDefense/PositiveFeedback.git
cd PositiveFeedback

conda create -n FP python=3.9
conda activate FP

pip install -r requirements.txt
pip install en_core_web_sm-3.7.1.tar.gz
pip install git+https://github.com/thunlp/OpenDelta.git
```

## Dataset
Please download the dataset from the following Google Drive link: https://drive.google.com/drive/folders/15Gd3tGc79pn6Q8Emvz-qj4pXGsb5LqXc.

It is recommended that the data files be organized in the following structure:
```yaml
datasets
└── retriever
    ├── nq
    │   ├── nq-train.jsonl
    │   └── nq-test.jsonl
    ├── hotpotqa
    │   ├── hotpotqa-train.jsonl
    │   └── hotpotqa-test.jsonl
    └── trivia
        ├── trivia-train.jsonl
        └── trivia-test.jsonl
```

## Data Format
The retrieval dataset is stored in the JSONL format, where each line corresponds to a single JSON-formatted data entry. \
Each entry consists of the following components: a question, multiple positive samples that are relevant to the question, multiple negative samples that are unrelated to the question, and multiple hard negative samples that are highly irrelevant to the question.
```json
{
    "dataset": "nq_test_psgs_w100",
    "question": "who sings does he love me with reba",
    "answers": [
        "Linda Davis"
    ],
    "positive_ctxs": [
        {
            "title": "Does He Love You",
            "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
            "score": 1000,
            "title_score": 1,
            "passage_id": "11828866"
        },
        ...
    ],
    "negative_ctxs": [
        {
            "title": "Cormac McCarthy",
            "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore \u2013 he quit 16 years ago in El Paso, with one of his young",
            "score": 0,
            "title_score": 0,
            "passage_id": "2145653"
        },
        {
            "title": "Pragmatic Sanction of 1549",
            "text": "one heir, Charles effectively united the Netherlands as one entity. After Charles' abdication in 1555, the Seventeen Provinces passed to his son, Philip II of Spain. The Pragmatic Sanction is said to be one example of the Habsburg contest with particularism that contributed to the Dutch Revolt. Each of the provinces had its own laws, customs and political practices. The new policy, imposed from the outside, angered many inhabitants, who viewed their provinces as distinct entities. It and other monarchical acts, such as the creation of bishoprics and promulgation of laws against heresy, stoked resentments, which fired the eruption of",
            "score": 0,
            "title_score": 0,
            "passage_id": "2271902"
        },
        ...
    ],
     "hard_negative_ctxs": [
        {
            "title": "Why Don't You Love Me (Beyonce\u0301 song)",
            "text": "song. According to the lyrics of \"Why Don't You Love Me\", Knowles impersonates a woman who questions her love interest about the reason for which he does not value her fabulousness, convincing him she's the best thing for him as she sings: \"Why don't you love me... when I make me so damn easy to love?... I got beauty... I got class... I got style and I got ass...\". The singer further tells her love interest that the decision not to choose her is \"entirely foolish\". Originally released as a pre-order bonus track on the deluxe edition of \"I Am...",
            "score": 14.678405,
            "title_score": 0,
            "passage_id": "14525568"
        },
        {
            "title": "Does He Love You",
            "text": "singing the second chorus. Reba stays behind the wall the whole time, while Linda is in front of her. It then briefly goes back to the dressing room, where Reba continues to smash her lover's picture. The next scene shows Reba approaching Linda's house in the pouring rain at night, while Linda stands on her porch as they sing the bridge. The scene then shifts to the next day, where Reba watches from afar as Linda and the man are seen on a speedboat, where he hugs her, implying that Linda is who he truly loves. Reba finally smiles at",
            "score": 14.385411,
            "title_score": 0,
            "passage_id": "11828871"
        },
        ...
    ]
}
```

## Pretrained Models
Download the models from Hugging Face using `huggingface-cli`:
```bash
huggingface-cli download google-bert/bert-base-uncased --local-dir common/pretrained-model/bert-base-uncased

huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir common/pretrained-model/bge-large-en-v1.5

huggingface-cli download WhereIsAI/UAE-Large-V1 --local-dir common/pretrained-model/uae-large-V1
```
Alternatively, you can use the provided script to download all models with a single command:
```bash
bash common/pretrained-model/download.sh
```

Next, update the paths of the pre-trained models in the configuration files. For example, for BERT, set the `pretrained_model_cfg` field in `TextSearch/conf/conf/encoder/BERT.yaml to common/pretrained-model/bert-base-uncased`.

For detailed information on the configuration structure and additional setup instructions, please refer to [`TextSearch/conf/README.md`](https://github.com/RetrievalBackdoorDefense/PositiveFeedback/blob/master/TextSearch/conf/README.md ).


## Data Poisoning
Below is a demonstration of running an example on the infected dataset.
```bash
python TextSearch/poison/main.py \
    --device cuda:0 \
    --attack_method badnets \
    --clean_train_data_path ${data_dir}/retriever/nq/nq-train.jsonl \
    --clean_test_data_path ${data_dir}/retriever/nq/nq-test.jsonl \
    --output_dir ${output_dir} \
    --dataname nq \
    --poison_rate 0.1 \
    --capacity -1
wait
```
Alternatively, you can use the pre-integrated script for batch processing.
```bash 
bash TextSearch/sh/poison.sh
```
### Parameters
Here is a description of the parameters:
- `device`: The device name to use (e.g., cuda:x).
- `attack_method`: The type of trigger to use. Choose from [badnets, addsent, stylebkd, hidden-killer].
- `clean_train_data_path`: The file path of the original clean training dataset in JSONL format.
- `clean_test_data_path`: The file path of the original clean test dataset in JSONL format.
- `output_dir`: The directory where the poisoned dataset will be saved.
- `dataname`: The identifier for the dataset. Choose from [nq, hotpotqa, trivia].
- `poison_rate`: The rate of poisoning in the dataset (e.g., 0.1, 0.05, 0.01).
- `capacity`: The size of the training set to use. -1 indicates using the entire dataset.


## Training
```bash
python TextSearch/train_dense_encoder.py \
    action=train \
    train_datasets=[nq_train,nq_poisoned_train_badnets_0.1] \
    test_datasets=[nq_test,nq_poisoned_test_badnets_0.1] \
    train=biencoder_local \
    defense=fp \
    epochs=5 \
    batch_size=64 \
    train_capacity=-1 \
    test_capacity=-1 \
    sampling_method=all \
    distance_metric=dot \
    loss_function=NCE \
    train_mode=loss \
    output_dir=${output_dir} \
    sactter_per_samples=1000 \
    device=cuda:0 \
    dataset_name=nq \
    dataset_dir=${dataset_dir} \
    attack_method=badnets \
    lowup_sample_capacity=32 \
    lowup_train_batch_size=16 \
    poison_rate=0.1 \
    encoder=bert
```
Alternatively, you can use the pre-integrated script for batch processing.
```bash
bash TextSearch/sh/run.sh
```
### Parameters
Here is a detailed description of the parameters:
- `train_datasets`: A list of identifiers for clean and poisoned training datasets.
Example: [nq_train, nq_poisoned_train_badnets_0.1].
- `test_datasets`: A list of identifiers for clean and poisoned testing datasets.
Example: [nq_dev, nq_poisoned_test_badnets_0.1].
- `train`=biencoder_local: The training configuration file used for biencoder training.
- `defense`: The defense method to apply. Options include [none, badacts, musclelora, onion, strip, cube, bki, fp].
- `train_capacity`: The number of training samples to use. -1 indicates using the entire training dataset.
- `test_capacity`: The number of testing samples to use. -1 indicates using the entire testing dataset.
- `distance_metric`: The distance metric used in the loss function.
Options: dot (dot product) or cosine (cosine similarity).
- `loss_function`: The loss function used during training. Default: NCE.
- `train_mode`: The training mode.
Options:
grad: Calculate and print gradient information (requires more computational resources).
loss: Calculate only the loss value.
output_dir: The directory where all model outputs will be saved.
- `scatter_per_samples`: The frequency of plotting points on the loss curve.
Example: Plot a point every scatter_per_samples steps.
- `device`: The device to use, specified in the format cuda:x.
- `dataset_name`: The name of the dataset.
Options: [nq, hotpotqa, trivia].
- `dataset_dir`: The directory where the dataset is stored.
- `attack_method`: The attack method (i.e., trigger type) to use.
Options: [badnets, addsent, hidden-killer, stylebkd].
- `lowup_sample_capacity`: The size of the low-up sample set (denoted as |Du|).
- `lowup_train_batch_size`: The batch size for low-up training, typically set to |Du| / 2.
- `poison_rate`: The rate of poisoning in the dataset.
- `encoder`: The pre-trained model to use for encoding.
