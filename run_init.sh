cd TinyRecursiveModels/
apt-get install -y vim gcc gpp
pip install -r ./requirements.txt
pip install torch
pip install wandb
pip install hydra-core
pip install adam-atan2-pytorch
pip install einops
pip install argdantic
pip install huggingface_hub
pip install visdom
wandb login 2957a68e9d48b6057e6cf86baa3f93c66cdb16d7

find /usr -name "libcuda.so*"
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1            /usr/lib/x86_64-linux-gnu/libcuda.so
sudo ldconfig
python3 -m dataset.build_arc_dataset   --input-file-prefix kaggle/combined/arc-agi   --output-dir data/arc2concept-aug-1000   --subsets training2 evaluation2 concept   --test-set-name evaluation2
python3 dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments
TORCH_NCCL_ENABLE_MONITORING=0 run_name="pretrain_mlp_t_sudoku" torchrun --nproc-per-node 2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.mlp_t=True arch.pos_encodings=none arch.L_layers=1 arch.H_cycles=3 arch.L_cycles=6 +run_name=${run_name} ema=True