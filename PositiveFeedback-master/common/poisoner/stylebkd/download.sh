# 'bible' 'shakespeare' 'twitter' 'lyrics' 'poetry'
styles=('poetry')

for style in "${styles[@]}"; do

model_dir=lievan/${style}
mkdir -p ${model_dir}

huggingface-cli download --resume-download lievan/${style} --local-dir ${model_dir}
wait

done