epochs=100
#learning_rate=0.001
scheduler_type=linear
accelerate launch training_veldt5_accelerate.py \
--finetune \
--vision_model 'KETI-AIR/veld-base' \
--language_model 'KETI-AIR/veld-base' \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--warmup_portion 0.02 \
--logging_steps 20 \
--checkpointing_steps 50000 \
--num_train_epochs $epochs \
--lr_scheduler_type $scheduler_type \
--with_tracking \
--output_dir model_output/veld_finetune_e${epochs}_${scheduler_type} \
--dataset_name_lm sent_dataset_bg.py \
--hf_data_dir_lm ../datasets/cc3m/ \
--hf_data_dir ../datasets/cc3m/images_384 \
--train_path ../datasets/cc3m/train_en_ko-filtered-0-0.05.json \
--validation_path ../datasets/cc3m/validation_en_ko-filtered.json \
--from_veld_pretrained 'KETI-AIR/veld-base' \
# --resume_from_checkpoint '../VL-KE-T5/model_output/veld_finetune_e100_linear/step_100000'

# epochs=10
# #learning_rate=0.001
# scheduler_type=linear
# accelerate launch training_veldt5_accelerate.py \
# --vision_model 'google/vit-base-patch16-384' \
# --language_model 'KETI-AIR/ke-t5-base' \
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --warmup_portion 0.02 \
# --logging_steps 20 \
# --checkpointing_steps 50000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir veld_finetune_e${epochs}_${scheduler_type} \
# --dataset_name_lm sent_dataset_bg.py \
# --hf_data_dir_lm ../datasets/cc3m/ \
# --hf_data_dir ../datasets/cc3m/images_384 \
# --train_path ../datasets/cc3m/train_en_ko-filtered.json \
# --validation_path ../datasets/cc3m/validation_en_ko-filtered.json \
# --from_veld_pretrained '../VL-KE-T5/VELD-pretrained/veld_e1_linear/'

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
# --hf_data_dir_lm ../datasets/cc3m/ \
# --hf_data_dir ../datasets/cc3m/images_384 \
# --train_path ../datasets/cc3m/train_en_ko-filtered.json \
# --validation_path ../datasets/cc3m/validation_en_ko-filtered.json \
