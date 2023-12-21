echo "=============================================================================="
echo "============ It runs training_retriever_accelerate"_bg".py ===================="
echo "=============================================================================="

epochs=3
learning_rate=0.001
scheduler_type=linear

accelerate launch training_retriever_accelerate_bg.py \
--image_root_dir ../datasets/cc3m/images_384 \
--train_path ../datasets/cc3m/train_en_ko-filtered.json \
--validation_path ../datasets/cc3m/validation_en_ko-filtered.json \
--vision_model 'google/vit-base-patch16-384' \
--language_model 'KETI-AIR/ke-t5-base' \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--warmup_portion 0.01 \
--learning_rate $learning_rate \
--logging_steps 20 \
--checkpointing_steps 1000 \
--num_train_epochs $epochs \
--lr_scheduler_type $scheduler_type \
--with_tracking \
--output_dir vl_norm_e${epochs}_${scheduler_type}_lr${learning_rate}
