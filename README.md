# Pytorch implementation [Self-Rule to Adapt (SRA)](https://openreview.net/forum?id=VO7asaS5GUk):
## Learning Generalized Features from Sparsely-Labeled Data Using Unsupervised Domain Adaptation for Colorectal Cancer Tissue Phenotyping.

Supervised learning is conditioned by the availability of labeled data, which are especially expensive to acquire in 
the field of medical image analysis. Making use of open-source data for pre-training or using domain adaptation can be 
a way to overcome this issue. However, pre-trained networks often fail to generalize to new test domains that are not 
distributed identically due to variations in tissue stainings, types, and textures. Additionally, current domain 
adaptation methods mainly rely on fully-labeled source datasets. In this work, we propose Self-Rule to Adapt (SRA)
which takes advantage of self-supervised learning to perform domain adaptation and removes the burden of fully-labeled 
source datasets. SRA can effectively transfer the discriminative knowledge obtained from a few labeled source domain to 
a new target domain without requiring additional tissue annotations. Our method harnesses both domains’ structures by
capturing visual similarity with intra-domain and cross-domain self-supervision. We show that our proposed method 
outperforms baselines across diverse domain adaptation settings and further validate our approach to our in-house 
clinical cohort.

![Segmentation result](figs/pipeline.png)

## Requirements
The implementation is an extension of the implementation of MoCoV2 ([paper](https://arxiv.org/abs/2003.04297), 
[code](https://github.com/facebookresearch/moco)).
 
Dataset:
* [Kather16](https://zenodo.org/record/53169): Collection of textures in colorectal cancer 
histology containing 5000 histological images
* [Kather19 - NCT-CRC-HE-100K](https://zenodo.org/record/1214456): 100,000 histological images of human colorectal cancer 
and healthy tissue

Python
* pytorch = 1.2.0
* torchvision = 0.4.0


## Usage
The pre-trained (Kather19 to Kather16) model is available on the google 
drive ([link](https://drive.google.com/drive/folders/1_4qa2JJPqMvEq6FgoTnmzkvPVgzQWma7?usp=sharing)). 

To train the model:
```bash
python train_sra.py --src_name kather19 --src_path /path/to/kather19 \
     --tar_name kather16 --tar_path /path/to/kather16 
```

To evaluate (generate embedding) and plot t-SNE projection:
```bash
python eval_sra.py --src_name kather19 --src_path /path/to/kather19 \
     --tar_name kather16 --tar_path /path/to/kather16 \
     --checkpoint /path/to/checkpoint.pth.tar
```

## Results

We present the t-SNE projection of the results of domain adaptation processes from Kather19 
to Kather16.
![Kather19 to Kather16](figs/tsne_k19k16.png)

To validate our approach on real case scenario, we perform domain adaptation using our 
proposed model from Kather19 to whole slide image 
sections from our in-house dataset. The results are presented here, alongside the original 
H&E image, their corresponding labels annotated by an expert pathologist, as well as 
comparative results of previous approaches smoothed using conditional random fields as 
in [L. Chan](https://github.com/lyndonchan/hsn_v1) (2018). The sections were selected such that, 
overall, they represent all tissue types equally.

![Segmentation result](figs/seg_wsi.png)

## Citation

If you use this work please use the following citation :).

```text
@inproceedings{
	abbet2021selfrule,
	title={Self-Rule to Adapt: Learning Generalized Features from Sparsely-Labeled Data Using Unsupervised Domain Adaptation for Colorectal Cancer Tissue Phenotyping},
	author={Christian Abbet and Linda Studer and Andreas Fischer and Heather Dawson and Inti Zlobec and Behzad Bozorgtabar and Jean-Philippe Thiran},
	booktitle={Medical Imaging with Deep Learning},
	year={2021},
	url={https://openreview.net/forum?id=VO7asaS5GUk}
}
```

