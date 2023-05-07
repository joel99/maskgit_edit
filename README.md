# MaskGIT-Edit

## Motivation
A full image synthesis workflow should allow user guidance at all levels of granularity. Text-based user inputs are suitable for coarse-grained control, but pixel-space markings are a more intuitive means for providing spatially precise user guidance. Note that in this regime, a user is typically fine-tuning small patches within a generated image, so our methods should primarily model the conditional generation of only the edited portions of the image.

Supervised tuning of a generative model to allow conditioning on user inputs is expensive as there is no true output matching a given user's edit in general. Fortuntaely, [SDEdit](https://sde-image-editing.github.io/) provides a proof of concept that clever interfacing with a generative model, without _any_ tuning, can integrate user edits quite well. However, the SDEdit model can only be formulated for diffusion models. User edit semantics are preserved only in so far as a diffusion process tends to not corrupt major edit structure. A diffusion edit method is convenient, but does not satisfy the rapid iteration requirement (for the standard diffusion model format), as the whole image is regenerated for a given edit. A candidate backbone that naturally fits the conditional generation problem is [MaskGIT](https://masked-generative-image-transformer.github.io/), which uses a masked autoencoding Vision Transformer over a diffusion process. There is no straightforward diffusion process to directly integrate edits, however, so we discuss how we approach the problem next.

## Approach

Pull slides.

### Implementation Note
We use the open-sourced MaskGIT repo -- unfortunately this only had an inference demo so effort was primarily directed to setting up a training pipeline once it seemed like tuning was inevitably needed. The open MaskGIT weights are only on ImageNet, so we use those.

## Comparisons



## Ablations



## Summary
