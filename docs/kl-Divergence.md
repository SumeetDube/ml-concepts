<iframe width="560" height="315" src="https://www.youtube.com/embed/q0AkK8aYbLY?si=yrj8iNVXqylaCNFa" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# KL-divergence
- Difference between two probability distributions
- ![[Pasted image 20240803184845.png]]
- The meric is Asymmetric.
- $Q$ - It is the reference/basis distribution, to which we want to compare change
- $P$ - The disitribution we want to compare change
- Averages are skewed by large number ie, 1 + 100000, won't give us a good discription
- We use log to solve averages skewed over larger number problem
- $P(x)$ is probabilistic weighting that prioritize Elements/variables according to how frequently they are occuring.

**Relative to $Q$ how much has $P$ changed
$$ D_{KL}(P||Q) = \sum_{x \in \mathcal{X} }P(X)\log\left(\frac{P(X)}{Q(X)}\right) $$

The Kullback-Leibler (KL) divergence is a measure of how one probability distribution differs from a second, reference probability distribution.
![[Pasted image 20240803190800.png]]
![[Pasted image 20240803190916.png]]

