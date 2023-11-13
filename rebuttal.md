# General Response

First of all, we would like to thank each reviewer for their 
constructive feedback.

## General comment about the significance or our experiments
Currently, the understanding of the empirical and certified 
robustness of the Perceptual metrics is very restricted, and
hence our work tries to demonstrate the importance of having a 
robust perceptual metric and provide a framework for training one. 
Based on the theoretical and empirical contributions,
we believe it is fair to claim that our work succeeded in 
(i) "demonstrating the threatens that come along with using non-robust
perceptual metrics" and (ii) "proposing a framework for training a 
robust perceptual metric with provable guarantees on any 2AFC
dataset (We show the results for NIGHT and BAPPS datasets that are 
the only available 2AFC datasets)(iii) "providing experiments that 
show the empirical robustness of LipSim metric in different settings 
than the 2AFC setting."

# Reviewer KFDY
Thank you for your thoughtful and positive comments.


**1- The presentation could be better. For example, the explanation of 2AFC is little which making it difficult to get the messages from this paper.** 

I will explain better in the final version 


**2- The experiments only conducted with Auto Attack. However, there are different kinds of attacks, and it would be good to experiment with other attack methods as well.**

I organize my response to the following points:

- We chose AutoAttack which is an empirically proven strong attack in the classification setting.
  
- Moreover, we have also the results for the $\ell_{\infty}$-AutoAttack  and $\ell_{\infty}$-PGD shown in Table 3 of the appendix.
- Additionally, we have performed $\ell_2$-PGD attack directly to the distance metric itself 
(which is different from classification settings) by optimizing the 
  $\underset{\delta}{argmax}\, d(x, x+\delta)$, and the results are shown in Figure 3b.
- In order to add more diversity, I added the Momentum Iterative
Attack for both $\ell_{\infty}$ and $\ell_{2}$. The results are as follows:
<table>
  <tr>
    <td rowspan="2" align="left">Metric</td>
    <td rowspan="2" align="center">Natural Score</td>
    <td colspan="3" align="center">$\ell_2$-MIA</td>
    <td colspan="3" align="center">$\ell_{\infty}$-MIA</td>
  </tr>
<tr>
<td align="center">0.5</td>
<td align="center">1.0</td>
<td align="center">2.0</td>
<td align="center">0.01</td>
<td align="center">0.02</td>
<td align="center">0.03</td>
</tr>
  <tr>
    <td>DreamSim</td>
    <td align="center">96.16</td>
<td align="center">61.79</td>
<td align="center">52.85</td>
<td align="center">52.69 </td>
<td align="center">2.08</td>
<td align="center">0.05</td>
<td align="center">0.0</td>
  </tr>
  <tr>
    <td align="center">LipSim</td>
    <td align="center">86.55</td>
<td align="center">82.79</td>
<td align="center">79.99</td>
<td align="center">80.10</td>
<td align="center">62.45</td>
<td align="center">34.38</td>
<td align="center">15.84</td>
  </tr>
</table>

**3- It would also be good to compare with other certified or non-certified defense methods.**

There exist two methods that are based on randomized smoothing. Besides the fact that 
they are computationally very expensive due to the Monte Carlo sampling 
for each data point, the proof requires the distance metric to hold symmetry property 
and triangle inequality. As perceptual metrics generally don't hold 
the triangle inequality, the triangle inequality approximation is 
used which makes the bound excessively loose.

# Reviewer YdcH
Thank you for your thoughtful and positive comments.

**1- The natural score of LipSim was observed to be lower than that of some competitors 
like DreamSim. This might raise concerns about its general performance when not 
under adversarial conditions.**

Figure 3.b provides a good big picture that lets us compare LipSim with different 
metrics in terms of natural score:

- Comparing LipSim with Low-level methods like PSNR and SSIM, 
we can see the superiority of LipSim.
- Comparing LipSim with the early, well-known perceptual metric, 
LPIPS, we find out that LipSim has better performance.
- Finally, comparing Lipsim with the SOTA perceptual metric/embeddings, 
we see LipSim is better than CLIP, which is a strong, weakly supervised 
representation learning method and although other methods are showing better 
performance in terms of natural score, their lack of robustness to 
tiny perturbations make a huge gap between robusntess and accuracy.
- On the other hand, the decent natural score of LipSim comes along with high 
robustness that makes the metric much more reliable.

  
**2- The real-world application testing of LipSim was primarily on image retrieval. 
It would be beneficial to see its performance on a wider variety of tasks.**

I can report the results for Instance Recognition and KNN.
# Reviewer sTNV

Thank you for your thoughtful and positive comments.
##

**1- Although there has been some previous works on robust perceptual 
metrics, authors claim theirs is the first one with provable 
guarantees. I still think discussing why this matters in practice 
is important.**

As mentioned in the paper, there exist previous works on the 
robustness of perceptual metrics but our work is the first to 
propose robust perceptual metrics with provable guarantees and 
with higher certificate radius. There exist two methods that are 
based on randomized smoothing. Besides the fact that they are 
computationally very expensive due to the Monte Carlo sampling 
for each data point, the proof requires the distance metric to 
hold symmetry property and triangle inequality. As perceptual 
metrics generally don't hold the triangle inequality, 
the triangle inequality approximation is used which makes 
the bound excessively loose.


**2- I see the application for image retrieval, but how can 
someone have access to the model (white box) to actually 
attack it. Please elaborate.**

The code is accessible through github and the instructions to easily 
loadin and using the metric is provided in the github repository.

**3- Consider changing the colors in bar plot of Fig. 3.a. They are hard to distinguish.**

Sure, the Fig. 3.b is changed and it's more visible now.

**4- Please mention why we need robust perceptual metrics??!!! In real world. In other words, how much impact such work has in real world and why it actually matters? An example may help here.**

Perceptual metrics have different applications in the real world, e.g. detecting 
images with any illegal content that shouldn't be posted and spread on social media.
if the perceptual metrics we are using are not robust, tiny perturbations 
could be crafted to mislead the model, and the illegal contents are not detected. 
In the paper we have provided examples in Figure 1. that show the huge perceptual 
mistake that DreamSim did while fed with the perturbed images with tiny perturbations.

**5- How about some non NN-based similarity metrics? I am guessing maybe there are not as good as NN-based ones but at least they might be more robust! Can you compare your method agains some of those m Metrics? Is DISTS in Fig. 3.a NN-based?**

We have reported the results for the non-NN-based metrics with the "Low level" title
in Figure 3.a.
The natural scores are very low and they are not comparable with the recent metrics. 
On the other hand, the pixel-wise methods are not capable of capturing the semantics 
of the image and are very vulnerable to tiny perturbations, e.g. shifting only one 
pixel can make the metric generate big numbers between the original and perturbed 
image.


**6- Other datasets than Nights dataset? What is the guarantee that these results will 
also generalize to other datasets?**

The 2AFC datasets are organized for training perceptual metrics. Currently two 2AFC
datasets are available: NIGHT and BAPPS. The result of LipSim on the NIGHT dataset
is reported in the paper. Here we report the result of LipSim on the BAPPS dataset. 
The LipSim model is not finetuned on the BAPPS dataset and the same model which is 
finetuned on the NIGHT dataset is used.
<table>
  <tr>
    <td rowspan="2" align="left">Metric</td>
    <td rowspan="2" align="center">Margin in Hinge Loss</td>
    <td rowspan="2" align="center">Natural Score</td>
    <td colspan="3" align="center">Certified Score</td>
  </tr>
  <tr>
    <td align="center">$\frac{36}{255}$</td>
    <td align="center">$\frac{72}{255}$</td>
    <td align="center">$\frac{108}{255}$</td>
  </tr>
  <tr>
    <td>DreamSim</td>
    <td align="center">-</td>
    <td align="center">78.47</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>

  </tr>
  <tr>
    <td align="center" rowspan="3">LipSim</td>
    <td align="center">0.2</td>
    <td align="center">73.47</td>
    <td align="center">30.0</td>
    <td align="center">12.96</td>
    <td align="center">5.33</td>
  </tr>
  <tr>
    <td align="center">0.4</td>
    <td align="center">74.31</td>
    <td align="center">31.74</td>
    <td align="center">15.19</td>
    <td align="center">7.0</td>
  </tr>
  <tr>
    <td align="center">0.5</td>
    <td align="center">74.29</td>
    <td align="center">31.20</td>
    <td align="center">15.07</td>
    <td align="center">6.77</td>
  </tr>
</table>

**7- In 5.1, you are stating “Can adversarial attacks against SOTA metrics cause: 
(1) misalignment with human perception?”. In section 4.1, you mention 
“Q: Please mention why we need robust perceptual metrics??!!! In real world. 
In other words, how much impact such work has in real world and why it actually 
matters? An example may help here. Alternatively, shouldn’t two image differ 
significantly when perturbation is high? Have you considered what happens at 
high perturbations? Shouldn’t a good perceptual metric match human judgements 
regardless of perturbation magnitude?**


##
**Table 2 shows the certified scores. However, the scores for 
perturbation 72/255 and beyond are still pretty low. 
How do humans behave in those perturbations?**

Human eye is completely robust to perturbed images with unbounded 
perturbation budgets. This paper is the first try to achieve 
certified robustness for perceptual metrics, and tried to take 
the first step by achieving certified robustness for 
tiny perturbations and the general robustness 
