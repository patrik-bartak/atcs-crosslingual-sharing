python finetune.py --savedir="train_res/xnli-new" --dataset=xnli --max_steps=200
python finetune.py --savedir="train_res/sib200-new" --dataset="Davlan/sib200" --max_steps=200 --batch_size=8

python finetune.py --savedir="train_res/sib200-new" --dataset="Davlan/sib200" --max_steps=200 --batch_size=8 --no-cuda
python finetune.py --savedir="train_res/xnli-new" --dataset=xnli --max_steps=200 --batch_size=8 --no-cuda
