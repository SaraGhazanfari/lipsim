# lipsim

# [LipSim: A Provably Robust Perceptual Similarity Metric](https://arxiv.org/abs/2310.18274)
## Abstract
<p align="justify"> In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an ℓ2 ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application.</p>

<img width="1452" alt="image" src="https://github.com/SaraGhazanfari/lipsim/assets/8003662/6e5660b5-f0ff-4330-bf9d-c34869f69d52">

## Certified Accuracy on 2AFC dataset (NIGHT)
<p align="justify"> Certified scores of LipSim given different settings. The natural and certified 2AFC scores of all variants of LipSim are shown in this figure. The LipSim - DreamSim version outperforms other variants regarding certified scores. The tradeoff between robustness and accuracy compares the results for different margins in the hinge loss. A higher margin parameter leads to a higher certified score and a lower natural score.</p>

<img width="500" align="center" alt="Certified_acc" src="https://github.com/SaraGhazanfari/lipsim/assets/8003662/f95b375b-5e0e-47e1-8b03-ae397fe44f8c">

## Empirical Accuracy on 2AFC dataset (NIGHT)
* <p align="justify"> Alignment on NIGHT dataset for original and perturbed images using AutoAttack. In this experiment, the perturbation is only applied on the reference images. While DreamSim employs an ensemble of three ViT-based models as the feature extractors, LipSim Backbone consists of a 1-Lipschitz network (composed of CNN and Linear layers) and is trained from scratch using the knowledge distillation approach.</p>

<img width="500" alt="L2" src="https://github.com/SaraGhazanfari/lipsim/assets/8003662/92293c29-03ad-4b60-9e58-cc41519d12d8">

* <p align="justify"> Figure a (left) compares percentages of alignment of several distance metrics with human vision based on the NIGHT dataset. As expected, the ViT-based methods outperform the pixel-wise and CNN-based metrics for the original images. However, LipSim with the 1-Lipschitz constraint backbone composed of CNN and Linear layers has a decent natural score and outperforms the (Base) Clip. Moreover, the figure shows the performance under attack (l2-AutoAttack with ε = 2.0) for the SOTA metric. While perturbing the reference image, other methods are experiencing a large decay in their performance but LipSim is showing much stronger robustness. Figure 3b (right) shows the distribution of d(x, x + δ) for LipSim and DreamSim. The δ perturbation is optimized for each method separately.</p>

<img width="1317" alt="results" src="https://github.com/SaraGhazanfari/lipsim/assets/8003662/d2e22c4b-7897-4414-a380-2c61fd19d364">

