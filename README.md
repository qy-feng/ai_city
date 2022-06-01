
## Setup

Install [miniconda](https://conda.io/en/latest/miniconda.html), then create the environment and activate it via

```sh
conda env create -f environment.yml
conda activate ai_city
```

## Evaluate

As a zero-shot system, no training is required.
We use [Mask R-CNN](http://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) pretrained on [COCO](http://cocodataset.org/#home) from [detectron2](https://github.com/facebookresearch/detectron2) as detector, whose weights will be downloaded automatically at the first run.

As the dataset only provided screenshots of the pre-defined routes, we created our own [annotation](monitor/tracks) of them with [labelme](https://github.com/wkentaro/labelme).

To get system outputs, run

python run.py **args

## Performance

Visualizations available at [Google Drive](https://drive.google.com/drive/folders/1s3TPykPa3JTaPOHUVOQF8S4iUi3SduAN?usp=sharing).

## Reference

```bib
@inproceedings{yu2020zero,
  title={Zero-VIRUS: Zero-shot VehIcle Route Understanding System for Intelligent Transportation},
  author={Yu, Lijun and Feng, Qianyu and Qian, Yijun and Liu, Wenhe and Hauptmann, Alexander G.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

https://github.com/open-mmlab/mmdetection

https://github.com/ZQPei/deep_sort_pytorch


