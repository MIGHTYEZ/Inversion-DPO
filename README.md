# <p align="center"> Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models </p>

<p align="center">
  <strong>🎉 Accepted at ACM MM 2025</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.11554">Paper Link</a> •
  <a href="https://huggingface.co/ezlee258258/Inversion-DPO">Huggingface Link</a>
</p>

<p align="center">
  by Zejian Li<sup>1</sup>, Yize Li<sup>1</sup>, Chenye Meng<sup>1</sup>, Zhongni Liu<sup>2</sup>, Ling Yang<sup>3</sup>, Shengyuan Zhang<sup>1</sup>, Guang Yang<sup>4</sup>, Changyuan Yang<sup>4</sup>, Zhiyuan Yang<sup>4</sup>, Lingyun Sun<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University  <sup>2</sup>University of Electronic Science and Technology of China  <sup>3</sup>Peking University  <sup>4</sup>Alibaba Group
</p>

![teaser](https://github.com/MIGHTYEZ/Inversion-DPO/blob/main/assets/pipeline.png)

## Abstract
Recent advancements in diffusion models (DMs) have been propelled by alignment methods that post-train models to better conform to human preferences. However, these approaches typically require computation-intensive training of a base model and a reward model, which not only incurs substantial computational overhead but may also compromise model accuracy and training efficiency. To address these limitations, we propose Inversion-DPO, a novel alignment framework that circumvents reward modeling by reformulating Direct Preference Optimization (DPO) with DDIM inversion for DMs. Our method conducts intractable posterior sampling in Diffusion-DPO with the deterministic inversion from winning and losing samples to noise and thus derive a new post-training paradigm. This paradigm eliminates the need for auxiliary reward models or inaccurate appromixation, significantly enhancing both precision and efficiency of training. We apply Inversion-DPO to a basic task of text-to-image generation and a challenging task of compositional image generation. Extensive experiments show substantial performance improvements achieved by Inversion-DPO compared to existing post-training methods and highlight the ability of the trained generative models to generate high-fidelity compositionally coherent images. For the post-training of compostitional image geneation, we curate a paired dataset consisting of 11,140 images with complex structural annotations and comprehensive scores, designed to enhance the compositional capabilities of generative models. Inversion-DPO explores a new avenue for efficient, high-precision alignment in diffusion models, advancing their applicability to complex realistic generation tasks.

![detail](https://github.com/MIGHTYEZ/Inversion-DPO/blob/main/assets/detail.png)


## Environment setup
The following commands are tested with Python 3.12.9 and CUDA 12.8.

Install required packages:

```
pip3 install -r requirements.txt
```
## Training
We provide a script for training Inversion-DPO. Use the following command to start:
```bash
chmod +x sdxl.sh
./sdxl.sh
```
## Inferencing
We provide a simple script to run inference using our publicly released weights on huggingface. You can easily get started by launching the `quick_samples.ipynb` Jupyter notebook.

## Model Weights
Our pretrained weights are available on [Hugging Face Hub](https://huggingface.co/ezlee258258/Inversion-DPO)


## Citation
```
@misc{li2025inversiondpo,
    title={Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models},
    author={Zejian Li and Yize Li and Chenye Meng and Zhongni Liu and Yang Ling and Shengyuan Zhang and Guang Yang and Changyuan Yang and Zhiyuan Yang and Lingyun Sun},
    year={2025},
    eprint={2507.11554},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement
This repository builds upon the [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) implementation. We sincerely appreciate the authors for their valuable contribution to the community.
