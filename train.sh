HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=6 python -u train.py --config-name="train_cls_arch" experiment=cls_arch run_id=OmniME
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=6 python motionfix_evaluate.py folder=/data1/home/Modelzhenwu1/OmniME/experiments/new_code/cls_arch/OmniME guidance_scale_text_n_motion=2.0 guidance_scale_motion=2.0 data=motionfix
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=6 python compute_metrics.py folder=/data1/home/Modelzhenwu1/OmniME/experiments/new_code/cls_arch/OmniME/3way_steps_300_motionfix_noise_last/ld_txt-2.0_ld_mot-2.0

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=5 python demo.py \
#     folder=/data2/home/Modelzhenwu1/SimMotionEdit_key_frame_triplet_id/experiments/new_code/cls_arch/idLoss-stance-real3 \
#     guidance_scale_text_n_motion=2.0 guidance_scale_motion=2.0 data=motionfix