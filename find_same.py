import joblib
import torch
from collections import defaultdict

def compare_motion_source(motion1, motion2):
    """比较两个 motion_source 是否完全相同"""
    if not isinstance(motion1, dict) or not isinstance(motion2, dict):
        return False
    if set(motion1.keys()) != set(motion2.keys()):
        return False
    
    required_keys = ['rots', 'trans', 'joint_positions']
    if set(motion1.keys()) != set(required_keys):
        return False
    
    # 检查形状
    if (motion1['rots'].shape != motion2['rots'].shape or
        motion1['trans'].shape != motion2['trans'].shape or
        motion1['joint_positions'].shape != motion2['joint_positions'].shape):
        return False
    
    # 检查值
    for key in required_keys:
        val1 = motion1[key]
        val2 = motion2[key]
        if not isinstance(val1, torch.Tensor) or not isinstance(val2, torch.Tensor):
            return False
        if not torch.equal(val1, val2):
            return False
    return True

def find_identical_motion_sources(file1, file2, file3):
    """找出三个文件中 motion_source 完全相同的样本 ID 组"""
    # 加载数据
    data1 = joblib.load(file1)
    data2 = joblib.load(file2)
    data3 = joblib.load(file3)
    
    # 存储所有样本：(来源文件, ID, motion_source)
    all_samples = []
    for id1 in data1:
        motion_source1 = data1[id1].get('motion_source')
        if motion_source1 is not None:
            all_samples.append(('data1', id1, motion_source1))
    for id2 in data2:
        motion_source2 = data2[id2].get('motion_source')
        if motion_source2 is not None:
            all_samples.append(('data2', id2, motion_source2))
    for id3 in data3:
        motion_source3 = data3[id3].get('motion_source')
        if motion_source3 is not None:
            all_samples.append(('data3', id3, motion_source3))
    
    # 分组：使用哈希表存储 motion_source 相同的 ID 组
    groups = defaultdict(list)
    processed = set()  # 记录已处理过的索引
    for i, (source_i, id_i, motion_i) in enumerate(all_samples):
        if i in processed:
            continue
        groups[(source_i, id_i)] = [(source_i, id_i)]  # 初始化组，包含自身
        processed.add(i)
        for j, (source_j, id_j, motion_j) in enumerate(all_samples[i+1:], i+1):
            if j not in processed and compare_motion_source(motion_i, motion_j):
                groups[(source_i, id_i)].append((source_j, id_j))
                processed.add(j)
    
    # 过滤出包含多个 ID 的组（忽略单样本组）
    identical_groups = [group for group in groups.values() if len(group) > 1]
    
    # 格式化输出：每组包含 ID 列表，标注来源
    formatted_groups = []
    for group in identical_groups:
        group_ids = [f"{source}:{id}" for source, id in group]
        formatted_groups.append(group_ids)
    
    return formatted_groups

# 示例用法
file1 = '/data2/home/Modelzhenwu1/SimMotionEdit_key_frame_triplet_id/data/motionfix-dataset-stance/motionfix_test.pth.tar'  # 替换为实际路径
file2 = '/data2/home/Modelzhenwu1/SimMotionEdit_key_frame_triplet_id/data/motionfix-dataset-stance/motionfix.pth.tar'  # 替换为实际路径
file3 = '/data2/home/Modelzhenwu1/SimMotionEdit_key_frame_triplet_id/data/motionfix-dataset-stance/motionfix.pth.tar'  # 替换为实际路径
identical_groups = find_identical_motion_sources(file1, file2, file3)
print("Identical motion_source groups:")
for group in identical_groups:
    print(group)