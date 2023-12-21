epochs=50
#learning_rate=0.001
scheduler_type=linear
# --main_process_port 29501
accelerate  launch training_veldt5_accelerate.py \
--finetune \
--vision_model 'google/vit-base-patch16-384' \
--language_model 'KETI-AIR/ke-t5-base' \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--warmup_portion 0.02 \
--logging_steps 20 \
--checkpointing_steps 50000 \
--num_train_epochs $epochs \
--lr_scheduler_type $scheduler_type \
--with_tracking \
--output_dir model_output/veld_finetune_coco_karparthy_start_not_aligned_e${epochs}_${scheduler_type} \
--dataset_name_lm sent_dataset_bg.py \
--hf_data_dir_lm ../datasets/cc3m/ \
--hf_data_dir ../datasets/coco-karparthy-complete/images_384/ \
--train_path ../datasets/coco-karparthy-complete/train_en_ko.json \
--validation_path ../datasets/coco-karparthy-complete/validation_en_ko.json \