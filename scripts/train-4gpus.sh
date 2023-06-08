#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
# export CUDA_VISIBLE_DEVICES=0

ID=2

## COCO
config_path="configs/COCO-InmodalSeg/aisformer_R_50_FPN_1x_modal_only_coco.yaml"

## kins
# config_path="configs/KINS-AmodalSeg/aisformer_R_50_FPN_1x_amodal_kins.yaml"
# config_path="configs/KINS-AmodalSeg/aisformer_R_50_FPN_1x_amodal_kins-4gpu.yaml"
# config_path="configs/KINS-AmodalSeg/aisformer_R_50_FPN_1x_modal_only_kins.yaml"

## d2sa
#config_path="configs/D2SA-AmodalSeg/aisformer_R_50_FPN_1x_amodal_d2sa.yaml"

## cocoa
#config_path="configs/COCOA_cls-AmodalSeg/aisformer_R_50_FPN_1x_amodal_cocoa_cls.yaml"

trap "kill 0" EXIT

# TORCH_DISTRIBUTED_DEBUG=DETAIL \
python3 tools/train_net.py --num-gpus 4 \
	--config-file ${config_path} 
	# 2>&1 | tee log/train_log_$ID.txt

wait

echo "############ Training is done! ############"
