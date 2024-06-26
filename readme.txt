# ComfyUI CFG+++ 

***this is a copy of this repo: https://gitea.com/NotEvilGirl/cfgpp ***


CFG++ implemented according to https://cfgpp-diffusion.github.io/
Basically modified DDIM sampler that makes sampling work at low CFG values (0 ~ 2)
Read the CFG++ paper for more details

"cfgpp_enabled" toggles the CFG++ sampling. When switched off it basically becomes DDIM sampler
"eta" makes sampler ancestral

!!!Warning!!!
That node is of alpha quality and next update might introduce breaking changes
!!!BUGS!!!
- CFG 1 breaks sampler with CFG++ enabled atm, set CFG to 1.01 or 0.99 to work it around
- CFG++ does not work with Stable Diffusion 3
