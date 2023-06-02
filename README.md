# Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata 
### John Kalkhof, Camila González, Anirban Mukhopadhyay
__https://arxiv.org/pdf/2302.03473.pdf__

<div class="center">
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Med-NCA performs segmentation with only 70k parameters by iterating the same local rule over each cell of an image. <a href="https://twitter.com/Hessian_AI?ref_src=twsrc%5Etfw">@Hessian_AI</a>, <a href="https://twitter.com/CS_TUDarmstadt?ref_src=twsrc%5Etfw">@CS_TUDarmstadt</a>, <a href="https://twitter.com/ipmi2023?ref_src=twsrc%5Etfw">@ipmi2023</a>, <a href="https://twitter.com/anirbanakash?ref_src=twsrc%5Etfw">@anirbanakash</a> <a href="https://t.co/N3umQy7Q0h">pic.twitter.com/N3umQy7Q0h</a></p>&mdash; John Kalkhof (@kalkjo) <a href="https://twitter.com/kalkjo/status/1638238606799040517?ref_src=twsrc%5Etfw">March 21, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</div>

> Access to the proper infrastructure is critical when performing medical image segmentation with Deep Learning. This requirement makes it difficult to run state-of-the-art segmentation models in resource-constrained scenarios like primary care facilities in rural areas and during crises. The recently emerging field of Neural Cellular Automata (NCA) has shown that locally interacting _one-cell_ models can achieve competitive results in tasks such as image generation or segmentations in low-resolution inputs. However, they are constrained by high VRAM requirements and the difficulty of reaching convergence for high-resolution images. To counteract these limitations we propose Med-NCA, an end-to-end NCA training pipeline for high-resolution image segmentation. Our method follows a two-step process. Global knowledge is first communicated between cells across the downscaled image. Following that, patch-based segmentation is performed. Our proposed Med-NCA outperforms the classic UNet by 2% and 3% Dice for hippocampus and prostate segmentation, respectively, while also being **500 times smaller**. We also show that Med-NCA is by design invariant with respect to image scale, shape and translation, experiencing only slight performance degradation even with strong shifts; and is robust against MRI acquisition artefacts. Med-NCA enables high-resolution medical image segmentation even on a Raspberry Pi B+, arguably the smallest device able to run PyTorch and that can be powered by a standard power bank.

<div>
<img src="/src/images/model_MedNCA.png" width="600"/>
</div>

To get started with this repository simply follow these few steps:

## Quickstart

1. Install requirements of repository: `pip install -r requirements.txt `
2. Download prostate dataset from: http://medicaldecathlon.com/
3. Adapt **img_path** and **label_path** in **train_Med_NCA.ipynb**
4. Run **train_Med_NCA.ipynb**
5. To view results in tensorboard: `tensorboard --logdir path`