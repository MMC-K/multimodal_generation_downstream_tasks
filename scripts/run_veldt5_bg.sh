epochs=10
#learning_rate=0.001
scheduler_type=linear
accelerate launch training_veldt5_accelerate.py \
--vision_model 'google/vit-base-patch16-384' \
--language_model 'KETI-AIR/ke-t5-base' \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--warmup_portion 0.02 \
--logging_steps 20 \
--checkpointing_steps 50000 \
--num_train_epochs $epochs \
--lr_scheduler_type $scheduler_type \
--with_tracking \
--output_dir veld_e${epochs}_${scheduler_type} \
--dataset_name_lm sent_dataset_bg.py \
--hf_data_dir_lm /mnt/nfs4/byunggill/datasets/cc3m/ \
--hf_data_dir /mnt/nfs4/byunggill/datasets/cc3m/images_384 \
--train_path /mnt/nfs4/byunggill/datasets/cc3m/train_en_ko-filtered.json \
--validation_path /mnt/nfs4/byunggill/datasets/cc3m/validation_en_ko-filtered.json \


# accelerate launch training_veldt5_accelerate.py \
# --vision_model 'google/vit-base-patch16-384' \
# --language_model 'KETI-AIR/ke-t5-base' \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --warmup_portion 0.02 \
# --logging_steps 20 \
# --checkpointing_steps 50000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir veld_e${epochs}_${scheduler_type} \
# --dataset_name_lm sent_dataset_bg.py \
# --hf_data_dir_lm /mnt/nfs4/byunggill/datasets/cc3m/ \
# --hf_data_dir /mnt/nfs4/byunggill/datasets/cc3m/images_384 \
# --train_path /mnt/nfs4/byunggill/datasets/cc3m/train_en_ko-filtered.json \
# --validation_path /mnt/nfs4/byunggill/datasets/cc3m/validation_en_ko-filtered.json \
