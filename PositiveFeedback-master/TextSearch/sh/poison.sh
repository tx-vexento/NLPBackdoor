export root_dir=""

data_dir=

#? cuda:x
device=cuda:1

#? nq hotpotqa trivia
dataset_names=(nq)
#? badnets addsent stylebkd hidden-killer
attack_methods=(badnets)
poison_rates=(0.1)

capacity=-1

for attack_method in "${attack_methods[@]}"; do
for dataset_name in "${dataset_names[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

output_dir=${data_dir}/retriever/${dataset_name}

cd ${root_dir}/poison
python main.py \
    --device ${device} \
    --attack_method ${attack_method} \
    --clean_train_data_path ${data_dir}/retriever/${dataset_name}/${dataset_name}-train.jsonl \
    --clean_test_data_path ${data_dir}/retriever/${dataset_name}/${dataset_name}-dev.jsonl \
    --output_dir  ${output_dir} \
    --dataname ${dataset_name} \
    --poison_rate ${poison_rate} \
    --capacity ${capacity}
wait

done
done
done