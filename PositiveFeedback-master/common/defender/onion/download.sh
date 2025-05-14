model_dir=./gpt2
mkdir -p ${model_dir}

modelscope download --model AI-ModelScope/gpt2 --local_dir ${model_dir}
# huggingface-cli download --resume-download openai-community/gpt2 --local-dir ${model_dir}
wait