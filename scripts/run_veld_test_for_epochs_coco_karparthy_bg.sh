ROOT_PATH=${1}

for model_path in ${ROOT_PATH}_*;
do
    echo "[*] executing bash ./scripts/run_veld_test_coco_karparthy_bg.sh ${model_path}"
    bash ./scripts/run_veld_test_coco_karparthy_bg.sh ${model_path} 2> /dev/null | tee ${model_path}/generate_result.txt
done