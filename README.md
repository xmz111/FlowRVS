<div align=“center” style=“font-family: charter;”>
<h1 align="center"> Deforming Videos to Masks: Flow Matching for Referring Video Segmentation </h1>

<p align="center">
  <a href='https://arxiv.org/abs/2510.06139'>
    <img src="https://img.shields.io/badge/arXiv%20paper-2510.06139-b31b1b.svg" alt="Paper">
  </a>
</p>

## 📢 News
[2026.01.26] FlowRVS was accepted by ICLR 2026!
[2026.02.05] We updated training codes.

## 🏄‍♂️ Overview


<p align="center">
  <img src="assets/begin_01.jpg" alt="Result" style="width:92%;">
  <figcaption style="text-align: center; margin-top: 10px; font-size: 0.95em;">
            <strong>FlowRVS</strong> replaces the cascaded ‘locate-then-segment’ paradigm (A) with a unified, end-to-end flow (B). This new paradigm avoids information bottlenecks, enabling superior handling of complex language and dynamic video (C) and achieving state-of-the-art performance (D).
        </figcaption>
</p>

**✨ Key Features:**

-   **FlowRVS** reformulates RVOS as learning a continuous, text-conditioned flow that deforms a video’s spatio-temporal representation into its target mask.
-   **FlowRVS** successfully  transfer the powerful text-to-video generative model to this RVOS task by proposing a suite of principled techniques.
-   **FlowRVS** achieves  new state of the art (SOTA) results on key benchmarks

<p align="center">
  <img src="assets/method.jpg" alt="Result" style="width:92%;">
</p>


## 🕒 Open-Source Plan
 - [x] Model and Inference Code
 - [x] Model Weight and Inference Guidance 
 - [x] Training Code and Guidance

## 🛠️ Environment Setup

#### 1. Create a conda environment
```
git clone https://github.com/xmz111/FlowRVS.git && cd FlowRVS
conda create -n flowrvs python=3.10 -y
conda activate flowrvs
```
#### 2. Install  dependencies
```
pip install -r requirements.txt
```

#### 3. Prepare Wan2.1 T2V model, we need config to construct models and T5 Encoder.
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-T2V-1.3B-Diffusers
```



## 🍻 Inference
### Inference on MeViS val and val_u splits.
#### 1. Prepare data
The dataset can be found in: https://github.com/henghuiding/MeViS  
After you successfully download the dataset, the file structure of the dataset should be like this:
* datasets
    * MeViS/
      * valid/
        * JPEGImages/  
        * meta_expressions.json
      * valid_u/
        * JPEGImages/
        * mask_dict.json
        * meta_expressions.json
```
pip install gdown
gdown https://drive.google.com/drive/folders/1MACaQ-O8seyMj-MBlycxRgCT08RVBZJp --folder -O dataset/MeViS/
```
#### 2. Download DiT and tuned VAE checkpoints from  https://huggingface.co/xmz111/FlowRVS  and place them as mevis_dit.pth and tuned_vae.pth;
#### 3.  Inference
Just run:

``` 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 inference_mevis.py --dit_ckpt=FlowRVS_dit_mevis.pth --vae_ckpt=tuned_vae.pth --output_dir=result --split=valid_u
```
   
Note that this code will cost about 33G GPU memory with default setting.


## 🥂 Training
Use --dataset_file to select training dataset (mevis, pretrain, ytvos), and use --resume to load checkpoint.
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  main.py  --dataset_file=mevis --num_frames=17 --lr=5e-5 --output_dir=mevis_training 
```
## 💚 Acknowledgement

We referenced the following works, and appreciate their contributions to the community.

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [MeViS](https://github.com/henghuiding/MeViS)


## 🔗 BibTeX
If you find our FlowRVS useful for your research and applications, please kindly cite us:
```
@article{wang2025flowrvs,
  title={Deforming Videos to Masks: Flow Matching for Referring Video Segmentation},
  author={Wang, Zanyi and Jiang, Dengyang and Li, Liuzhuozheng and Dang, Sizhe and Li, Chengzu and Yang, Harry and Dai, Guang and Wang, Mengmeng and Wang, Jingdong},
  journal={arXiv preprint arXiv:2510.06139}, 
  year={2025}
}
```


