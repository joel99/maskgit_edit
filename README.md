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

Thus, MaskGIT does not by default condition in $E$ to sample $X_E$. We must either tune the model to do so, or heuristically hack the inference process. In either case, we split the conditioning into two pieces: for a particular $X_{j \in E}$, can we use $G_j$ (self-guidance) and $G_{E \setminus j}$ (context-guidance)?

**Non-tuning Heuristics**
- Context-guidance can be used by manipulating inference. At each step of ancestral sampling, we replace the previous round's least confident samples with guidance. (See `maskgit/libml/parallel_decode.decode_context_guidance`)
- Self-guidance can reweight sampled confidences via a pre-specified distance metric. I use L2.

**Tuning**
- Both forms of guidance might be achieved if the model can be trained to use low confidence samples -- such as those produced in its own iterative sampling. I implement this by retaining the low-confidence samplesd tokens from an initial iteration and adding their embedding to another iteration (See `maskgit.nets.maskgit_transformer`). Guidance can then be provided at test-time identically to how it's provided in modified training.
- Learned reweighting for self-guidance: by learning a thin reweighting matrix of size $C \times C$ (representing affinities between the $C$ codes), we can attempt to learn self-guidance without worrying about overfit in the Transformer.


### Implementation Note
We use the open-sourced MaskGIT repo -- unfortunately this only had an inference demo so much effort was directed to setting up a training pipeline; this kind of killed most progress on the project.

The open MaskGIT weights are only on ImageNet, so we use those.

## Results



## Comparisons and Ablations



## Summary
Motivationally, the spatial benefits of MaskGIT are entirely complementary to SDEdit's iterative denoising. Uncertainty is a natural way of trading off fidelity to user guidance with realism.

This project was technically very difficult