DATASET0="TEMPERATURE0"
DATASET1="TEMPERATURE1"
DATASET2="TEMPERATURE2"

python3 maf.py --train --model realnvp --dataset $DATASET0 --output_dir $DATASET0 --conditional --log_interval 10 --batch_size 20 --n_epochs 50 --lr 0.001
python3 maf.py --train --model realnvp --dataset $DATASET1 --output_dir $DATASET1 --conditional --log_interval 10 --batch_size 20 --n_epochs 50 --lr 0.001
python3 maf.py --train --model realnvp --dataset $DATASET2 --output_dir $DATASET2 --conditional --log_interval 10 --batch_size 20 --n_epochs 50 --lr 0.001
python3 maf.py --kl --model realnvp --dataset $DATASET0 --conditional --restore_file $DATASET0/best_model_checkpoint.pt --restore_file_compare $DATASET1/best_model_checkpoint.pt
python3 maf.py --kl --model realnvp --dataset $DATASET0 --conditional --restore_file $DATASET0/best_model_checkpoint.pt --restore_file_compare $DATASET2/best_model_checkpoint.pt
