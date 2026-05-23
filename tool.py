import joblib

# 加载 .pth.tar 文件
data = joblib.load("/data1/home/Modelzhenwu1/SimMotionEdit_key_frame_triplet_id/data/motionfix-dataset/motionfix.pth.tar")
i = 0
# 遍历每个样本
for sample_name, sample_data in data.items():
    motion_source = sample_data.get('motion_source')
    motion_target = sample_data.get('motion_target')
    
    if motion_source is None or motion_target is None:
        print(00000000)
        continue
    
    # 检查 T 维度是否相等
    if motion_source["trans"].shape[0] != motion_target["trans"].shape[0]:
        i += 1
print(i)       
