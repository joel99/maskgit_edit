# MaskGIT-Edit

## Motivation
A full image synthesis workflow should allow user guidance at all levels of granularity. Text-based user inputs are suitable for coarse-grained control, but pixel-space markings are a more intuitive means for providing spatially precise user guidance. Note that in this regime, a user is typically fine-tuning small patches within a generated image, so our methods should primarily model the conditional generation of only the edited portions of the image.

Supervised tuning of a generative model to allow conditioning on user inputs is expensive as there is no true output matching a given user's edit in general. Fortuntaely, [SDEdit](https://sde-image-editing.github.io/) provides a proof of concept that clever interfacing with a generative model, without _any_ tuning, can integrate user edits quite well. However, the SDEdit model can only be formulated for diffusion models. User edit semantics are preserved only in so far as a diffusion process tends to not corrupt major edit structure. A diffusion edit method is convenient, but does not satisfy the rapid iteration requirement (for the standard diffusion model format), as the whole image is regenerated for a given edit. A candidate backbone that naturally fits the conditional generation problem is [MaskGIT](https://masked-generative-image-transformer.github.io/), which uses a masked autoencoding Vision Transformer rather than a diffusion process. There is no straightforward diffusion process to directly integrate edits, however, so we discuss how we approach the problem next.

## Approach

The problem formalizes as follows: denote flattened image pixels or patches as $X_{[n]}$, and denote guidance as $G_{E}, E \subset [n]$. Our guided image is thus $X^{(g)} = \{X_{[n]\setminus E}, G_{E}\}$.  Then, we want to use a generative model's pretrained
$$
    p_\theta (X_E | X_{[n] \setminus E})
$$
To model
$$
    p (X_E | X_{[n] \setminus E}, G_E)
$$
SDEdit bridges the gap between the two distributions with forward diffusion: under sufficient noise, $G_E$ is indistinguishable from $X_E$. So they sampled a noised guidance
$
X^{(g)}(t_0) \sim \mathcal{N}(X^{(g)}; \sigma^2 (t_0) I)
$, and then directly apply the pretrained reverse diffusion.
Uncertainty is a great way to fold in user edits and future methods will likely retain the idea; MaskGIT does _not_ pretrain with uncertain inputs, but does let us focus on spatial dependencies. Let's investigate.

First, MaskGIT is a Vector-Quantized Transformer that first encodes image patches into discrete codes. This project, like MaskGIT itself, focuses on the Transformer generative model over code tokens, not pixels. MaskGIT uses beam-search/ancestral sampling to draw from $p(X_E|X_{[n]\setminus E})$ over $K$ steps; each step conditions _only on previously sampled tokens_ and locks in the most confident tokens.

Thus, MaskGIT does not by default condition in $E$ to sample $X_E$. We must reframe the inference process. We split this into two steps: when sampling a particular token $X_{j \in E}$, can we use $G_j$ (self-guidance) and $G_{E \setminus j}$ (context-guidance)?

- Context-guidance can be used through an inference only manipulation. We retain confidences from the previous sampling step and replace the lowest

1. Patching and ancestral sampling.
2. Self-conditioning.



### Implementation Note
We use the open-sourced MaskGIT repo -- unfortunately this only had an inference demo so effort was primarily directed to setting up a training pipeline once it seemed like tuning was inevitably needed. The open MaskGIT weights are only on ImageNet, so we use those.

## Comparisons



## Ablations



## Summary
Motivationally, the spatial benefits of MaskGIT are entirely complementary to SDEdit's iterative denoising. Uncertainty is a natural way of trading off fidelity to user guidance with realism.