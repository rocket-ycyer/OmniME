<p align="center">
  <img src="cover.png" width="100%" alt="OmniME Cover">
</p>

<h1 align="center">Omni-Supervised Motion Editing: Balancing Change and Invariance through Positive-Negative Learning</h1>

<p align="center">
  CVPR 2026
</p>

<p align="center">
  <a href='https://github.com/rocket-ycyer/OmniME'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white' alt='GitHub'>
  </a>
</p>

---

## Environment & Data Setup

Our data and environment follow [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git). Please refer to [motionfix](https://github.com/atnikos/motionfix) to download the dataset and set up the environment, then place the data in the corresponding locations.

## Pretrained Checkpoint

You can download our pretrained checkpoint from [Baidu Pan](https://pan.baidu.com/s/1Ek0h_I-EKEDY8h6FaBRk8Q?pwd=sd9s) (code: `sd9s`). The directory structure follows the same layout as [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git).

## Training

```bash
python -u train.py --config-name="train_cls_arch" experiment=cls_arch run_id=OmniME
```

## Evaluation

#### Step 1: Extract samples

```bash
python motionfix_evaluate.py \
    folder=/path/to/exp \
    guidance_scale_text_n_motion=2.0 \
    guidance_scale_motion=2.0 \
    data=motionfix
```

#### Step 2: Compute metrics

```bash
python compute_metrics.py folder=/path/to/exp/samples/npys
```

## Demo

```bash
python demo.py \
    folder=/path/to/exp \
    guidance_scale_text_n_motion=2.0 \
    guidance_scale_motion=2.0 \
    data=motionfix
```

## Acknowledgements

Our code is based on [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit.git) and [motionfix](https://github.com/atnikos/motionfix).

## License

This code is distributed under an MIT LICENSE. We also include the LICENSE of motionfix in this repo. Other third-party datasets and software are subject to their respective licenses.

## Citation

```bibtex
@inproceedings{shi2026omnime,
  title={Omni-Supervised Motion Editing: Balancing Change and Invariance through Positive-Negative Learning},
  author={Zhenwu Shi and Jingyu Gong and Wenxi Li and Yuan Fang and Peiwei Wang and Xingzan Wang and Qian Tianwen and Jiao Xie and Lizhuang Ma and Shaohui Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
