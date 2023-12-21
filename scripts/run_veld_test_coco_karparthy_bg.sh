#MODEL_PATH="../VL-KE-T5/model_output/veld_finetune_e100_linear_27/"
MODEL_PATH=${1}
if [[ $# -eq 0 ]];
then
MODEL_PATH="KETI-AIR/veld-base"
fi
echo "[*] MODEL_PATH: "${MODEL_PATH}
# accelerate launch testing_veldt5_partial_state_bg.py \
# accelerate launch testing_veldt5_accelerate_bg.py \
accelerate launch  testing_veldt5_accelerate_bg.py \
--save_caption_result \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--hf_data_dir ../datasets/coco-karparthy-complete/images_384/ \
--validation_path ../datasets/coco-karparthy-complete/validation_en_ko.json \
--vision_model ${MODEL_PATH} \
--language_model ${MODEL_PATH} \
--from_veld_model ${MODEL_PATH}


# --vision_model '../VL-KE-T5/veld_finetune_e1_linear_0/' \
# --language_model '../VL-KE-T5/veld_finetune_e1_linear_0/' \
# --from_veld_model "../VL-KE-T5/veld_finetune_e1_linear_0/"

# --vision_model '../VL-KE-T5/veld_finetune_start_not_aligned_e1_linear_0/' \
# --language_model '../VL-KE-T5/veld_finetune_start_not_aligned_e1_linear_0/' \
# --from_veld_model "../VL-KE-T5/veld_finetune_start_not_aligned_e1_linear_0/"

# --vision_model '../VL-KE-T5/VELD-pretrained/veld_e1_linear/' \
# --language_model '../VL-KE-T5/VELD-pretrained/veld_e1_linear/' \
# --from_veld_model "../VL-KE-T5/VELD-pretrained/veld_e1_linear/"

# --vision_model 'KETI-AIR/veld-base' \
# --language_model 'KETI-AIR/veld-base' \
# --from_veld_model "KETI-AIR/veld-base"


#
# \

# --vision_model and --language_model are provided for tokenizers.

# --warmup_portion 0.02 \
# --logging_steps 20 \
# --checkpointing_steps 50000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir ____veld_finetune_e${epochs}_${scheduler_type} \
# --dataset_name_lm sent_dataset_bg.py \
# --train_path ../datasets/cc3m/train_en_ko-filtered.json \