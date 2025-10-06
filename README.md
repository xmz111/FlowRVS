<div align=“center” style=“font-family: charter;”>
<h1 align="center"> Deforming Videos to Masks: Flow Matching for Referring Video Segmentation </h1>
<div align="center">
  <a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
</div>



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


## 🔥 News

- __[2025.10.12]__:  🎉 
- __[2025.10.12]__:  🎉 
- __[2025.10.8]__:   🎉 

## 🕒 Open-source Plan
- Inference codes are uploaded.
- Other codes and weight swill come soon.

## 🛠️ Environment Setup
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-T2V-1.3B-Diffusers
```
Prepare Wan2.1 T2V model first.





## 🍻 Inference



## 🥂 Training


## 💚 Acknowledgement

We referenced the following works, and appreciate their contributions to the community.

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)


## 🔗 BibTeX
If you find our FlowRVS useful for your research and applications, please kindly cite us:
```

```


