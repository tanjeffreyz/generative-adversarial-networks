<h1 align="center">Generative Adversarial Networks</h1>

PyTorch implementation of "Generative Adversarial Networks" by Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, 
Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.


## MNIST Results

<table><tr>
    <td>
        <img src="models/mnist/05_16_2022/17_05_14/samples/final.png" />
    </td>
    <td>
        <img src="models/mnist/05_16_2022/17_05_14/metrics.png" />
    </td>
</tr></table>


<details><summary><b>Notable Failures</b></summary>

### Modal Collapse
<table><tr>
    <td>
        <img src="models/mnist/04_23_2022/20_48_49_modal_collapse/samples/final.png" />
    </td>
    <td>
        <img src="models/mnist/04_23_2022/20_48_49_modal_collapse/metrics.png" />
    </td>
</tr></table>


### Convergence Fail
<table><tr>
    <td>
        <img src="models/mnist/04_23_2022/21_39_37_convergence_fail/samples/final.png" />
    </td>
    <td>
        <img src="models/mnist/04_23_2022/21_39_37_convergence_fail/metrics.png" />
    </td>
</tr></table>


### Discriminator using Batch Normalization
<table><tr>
    <td>
        <img src="models/mnist/05_14_2022/13_00_31_d_batchnorm/samples/final.png" />
    </td>
    <td>
        <img src="models/mnist/05_14_2022/13_00_31_d_batchnorm/metrics.png" />
    </td>
</tr></table>

</details>









## CIFAR-10 Results











## Notes
### Proof of Optimality
The handwritten math below shows the work that was omitted in [1].
![](images/optimality.png)

### Kullback-Leibler Divergence (KL)
According to [2], KL is also known as **relative entropy**. It measures how much one probability distribution differs from another distribution.

![](images/kullback_leibler.png)

The Kullback-Leibler divergence can also be viewed as **excess entropy**, which is the amount of
extra information that must be communicated for a code that is optimal for **_Q_** but not for **_P_**, compared to a code that
is optimal for **_P_**.

![](images/kullback_leibler_motivation.png)


### Jensen-Shannon Divergence (JSD)
The Jensen-Shannon divergence is defined in [3] as:

![](images/jensen_shannon.png) &emsp; where &emsp; ![](images/jsd_m.png)

**Properties**:
- JSD is non-negative
- JSD is symmetric: `JSD(P || Q) = JSD(Q || P)`
- **JSD is 0 iff P = Q**




## References
[[1](https://arxiv.org/abs/1406.2661)] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville, Yoshua Bengio. _Generative Adversarial Networks_. 
arXiv:1406.2661v1 [stat.ML] 10 Jun 2014

[[2](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)] Wikipedia. _Kullback–Leibler divergence_.

[[3](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)] Wikipedia. _Jensen–Shannon divergence_.

