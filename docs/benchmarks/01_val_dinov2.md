# Benchmark report: Community Forensics

- Entries evaluated: 1000
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| constant_fake | fixed | 0.500 | 0.376 | 0.376 | 0.500 | 1.000 | 1.000 | 0.624 | 0.624 |
| constant_fake | tuned | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.624 | 0.624 |
| constant_real | fixed | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.376 | 0.376 |
| constant_real | tuned | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.376 | 0.376 |
| dinov2_head | fixed | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.001 | 0.000 |
| dinov2_head | tuned | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.001 | 0.000 |
| random | fixed | 0.492 | 0.375 | 0.491 | 0.491 | 0.510 | 0.492 | 0.286 | 0.339 |
| random | tuned | 0.492 | 0.375 | 0.398 | 0.511 | 0.946 | 0.968 | 0.286 | 0.339 |
| signals_mean | fixed | 0.341 | 0.357 | 0.320 | 0.410 | 0.954 | 0.774 | 0.159 | 0.261 |
| signals_mean | tuned | 0.341 | 0.357 | 0.636 | 0.523 | 0.021 | 0.066 | 0.159 | 0.261 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| constant_fake | clean | 0.500 | 0.500 |
| constant_fake | jpeg_q95 | 0.500 | 0.500 |
| constant_fake | jpeg_q85 | 0.500 | 0.500 |
| constant_fake | jpeg_q75 | 0.500 | 0.500 |
| constant_fake | jpeg_q60 | 0.500 | 0.500 |
| constant_fake | jpeg_q50 | 0.500 | 0.500 |
| constant_fake | webp_q80 | 0.500 | 0.500 |
| constant_fake | resize_0.75 | 0.500 | 0.500 |
| constant_fake | resize_0.5 | 0.500 | 0.500 |
| constant_fake | resize_0.25 | 0.500 | 0.500 |
| constant_fake | roundtrip_0.5 | 0.500 | 0.500 |
| constant_fake | crop_0.8 | 0.500 | 0.500 |
| constant_fake | noise_2 | 0.500 | 0.500 |
| constant_fake | noise_5 | 0.500 | 0.500 |
| constant_fake | social_1080_q80 | 0.500 | 0.500 |
| constant_real | clean | 0.500 | 0.500 |
| constant_real | jpeg_q95 | 0.500 | 0.500 |
| constant_real | jpeg_q85 | 0.500 | 0.500 |
| constant_real | jpeg_q75 | 0.500 | 0.500 |
| constant_real | jpeg_q60 | 0.500 | 0.500 |
| constant_real | jpeg_q50 | 0.500 | 0.500 |
| constant_real | webp_q80 | 0.500 | 0.500 |
| constant_real | resize_0.75 | 0.500 | 0.500 |
| constant_real | resize_0.5 | 0.500 | 0.500 |
| constant_real | resize_0.25 | 0.500 | 0.500 |
| constant_real | roundtrip_0.5 | 0.500 | 0.500 |
| constant_real | crop_0.8 | 0.500 | 0.500 |
| constant_real | noise_2 | 0.500 | 0.500 |
| constant_real | noise_5 | 0.500 | 0.500 |
| constant_real | social_1080_q80 | 0.500 | 0.500 |
| dinov2_head | clean | 1.000 | 1.000 |
| dinov2_head | jpeg_q95 | 1.000 | 1.000 |
| dinov2_head | jpeg_q85 | 1.000 | 1.000 |
| dinov2_head | jpeg_q75 | 1.000 | 1.000 |
| dinov2_head | jpeg_q60 | 1.000 | 1.000 |
| dinov2_head | jpeg_q50 | 1.000 | 1.000 |
| dinov2_head | webp_q80 | 1.000 | 1.000 |
| dinov2_head | resize_0.75 | 1.000 | 1.000 |
| dinov2_head | resize_0.5 | 0.998 | 0.947 |
| dinov2_head | resize_0.25 | 0.567 | 0.500 |
| dinov2_head | roundtrip_0.5 | 1.000 | 1.000 |
| dinov2_head | crop_0.8 | 1.000 | 1.000 |
| dinov2_head | noise_2 | 1.000 | 1.000 |
| dinov2_head | noise_5 | 1.000 | 1.000 |
| dinov2_head | social_1080_q80 | 1.000 | 1.000 |
| random | clean | 0.492 | 0.491 |
| random | jpeg_q95 | 0.518 | 0.513 |
| random | jpeg_q85 | 0.512 | 0.514 |
| random | jpeg_q75 | 0.497 | 0.489 |
| random | jpeg_q60 | 0.457 | 0.477 |
| random | jpeg_q50 | 0.506 | 0.504 |
| random | webp_q80 | 0.501 | 0.489 |
| random | resize_0.75 | 0.482 | 0.477 |
| random | resize_0.5 | 0.502 | 0.508 |
| random | resize_0.25 | 0.500 | 0.506 |
| random | roundtrip_0.5 | 0.506 | 0.503 |
| random | crop_0.8 | 0.517 | 0.516 |
| random | noise_2 | 0.536 | 0.543 |
| random | noise_5 | 0.500 | 0.507 |
| random | social_1080_q80 | 0.497 | 0.493 |
| signals_mean | clean | 0.341 | 0.410 |
| signals_mean | jpeg_q95 | 0.480 | 0.484 |
| signals_mean | jpeg_q85 | 0.298 | 0.394 |
| signals_mean | jpeg_q75 | 0.334 | 0.381 |
| signals_mean | jpeg_q60 | 0.390 | 0.418 |
| signals_mean | jpeg_q50 | 0.390 | 0.408 |
| signals_mean | webp_q80 | 0.547 | 0.476 |
| signals_mean | resize_0.75 | 0.310 | 0.415 |
| signals_mean | resize_0.5 | 0.358 | 0.432 |
| signals_mean | resize_0.25 | 0.473 | 0.477 |
| signals_mean | roundtrip_0.5 | 0.307 | 0.464 |
| signals_mean | crop_0.8 | 0.350 | 0.433 |
| signals_mean | noise_2 | 0.306 | 0.425 |
| signals_mean | noise_5 | 0.220 | 0.477 |
| signals_mean | social_1080_q80 | 0.322 | 0.373 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| constant_fake | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| constant_fake | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| constant_fake | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| constant_fake | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| constant_fake | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| constant_fake | Masagin/Deliberate | only one label present; AUC undefined |
| constant_fake | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| constant_fake | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| constant_fake | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| constant_fake | Yntec/CultClassic | only one label present; AUC undefined |
| constant_fake | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| constant_fake | Yntec/LehinaModel | only one label present; AUC undefined |
| constant_fake | Yntec/Paragon | only one label present; AUC undefined |
| constant_fake | Yntec/StolenDreams | only one label present; AUC undefined |
| constant_fake | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| constant_fake | Yntec/fennPhoto | only one label present; AUC undefined |
| constant_fake | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| constant_fake | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| constant_fake | briannlongzhao/0 | only one label present; AUC undefined |
| constant_fake | briannlongzhao/2 | only one label present; AUC undefined |
| constant_fake | digiplay/Realisian_v1 | only one label present; AUC undefined |
| constant_fake | digiplay/Remedy | only one label present; AUC undefined |
| constant_fake | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| constant_fake | employelalisa/min | only one label present; AUC undefined |
| constant_fake | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| constant_fake | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| constant_fake | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| constant_fake | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| constant_fake | lewdryuna/A-Rainier | only one label present; AUC undefined |
| constant_fake | livingbox/modern-style-v4 | only one label present; AUC undefined |
| constant_fake | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| constant_fake | none | only one label present; AUC undefined |
| constant_fake | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| constant_fake | openskyml/midjourney-mini | only one label present; AUC undefined |
| constant_fake | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| constant_fake | plasmo/woolitize | only one label present; AUC undefined |
| constant_fake | prushton/text-inv-myra | only one label present; AUC undefined |
| constant_fake | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| constant_fake | segmind/SSD-1B | only one label present; AUC undefined |
| constant_fake | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| constant_fake | songhee/rugged-sporty-car | only one label present; AUC undefined |
| constant_fake | steb6/textual_inversion_cat | only one label present; AUC undefined |
| constant_fake | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| constant_real | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| constant_real | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| constant_real | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| constant_real | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| constant_real | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| constant_real | Masagin/Deliberate | only one label present; AUC undefined |
| constant_real | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| constant_real | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| constant_real | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| constant_real | Yntec/CultClassic | only one label present; AUC undefined |
| constant_real | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| constant_real | Yntec/LehinaModel | only one label present; AUC undefined |
| constant_real | Yntec/Paragon | only one label present; AUC undefined |
| constant_real | Yntec/StolenDreams | only one label present; AUC undefined |
| constant_real | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| constant_real | Yntec/fennPhoto | only one label present; AUC undefined |
| constant_real | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| constant_real | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| constant_real | briannlongzhao/0 | only one label present; AUC undefined |
| constant_real | briannlongzhao/2 | only one label present; AUC undefined |
| constant_real | digiplay/Realisian_v1 | only one label present; AUC undefined |
| constant_real | digiplay/Remedy | only one label present; AUC undefined |
| constant_real | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| constant_real | employelalisa/min | only one label present; AUC undefined |
| constant_real | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| constant_real | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| constant_real | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| constant_real | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| constant_real | lewdryuna/A-Rainier | only one label present; AUC undefined |
| constant_real | livingbox/modern-style-v4 | only one label present; AUC undefined |
| constant_real | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| constant_real | none | only one label present; AUC undefined |
| constant_real | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| constant_real | openskyml/midjourney-mini | only one label present; AUC undefined |
| constant_real | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| constant_real | plasmo/woolitize | only one label present; AUC undefined |
| constant_real | prushton/text-inv-myra | only one label present; AUC undefined |
| constant_real | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| constant_real | segmind/SSD-1B | only one label present; AUC undefined |
| constant_real | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| constant_real | songhee/rugged-sporty-car | only one label present; AUC undefined |
| constant_real | steb6/textual_inversion_cat | only one label present; AUC undefined |
| constant_real | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| dinov2_head | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| dinov2_head | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| dinov2_head | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| dinov2_head | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| dinov2_head | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| dinov2_head | Masagin/Deliberate | only one label present; AUC undefined |
| dinov2_head | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| dinov2_head | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| dinov2_head | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| dinov2_head | Yntec/CultClassic | only one label present; AUC undefined |
| dinov2_head | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| dinov2_head | Yntec/LehinaModel | only one label present; AUC undefined |
| dinov2_head | Yntec/Paragon | only one label present; AUC undefined |
| dinov2_head | Yntec/StolenDreams | only one label present; AUC undefined |
| dinov2_head | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| dinov2_head | Yntec/fennPhoto | only one label present; AUC undefined |
| dinov2_head | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| dinov2_head | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| dinov2_head | briannlongzhao/0 | only one label present; AUC undefined |
| dinov2_head | briannlongzhao/2 | only one label present; AUC undefined |
| dinov2_head | digiplay/Realisian_v1 | only one label present; AUC undefined |
| dinov2_head | digiplay/Remedy | only one label present; AUC undefined |
| dinov2_head | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| dinov2_head | employelalisa/min | only one label present; AUC undefined |
| dinov2_head | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| dinov2_head | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| dinov2_head | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| dinov2_head | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| dinov2_head | lewdryuna/A-Rainier | only one label present; AUC undefined |
| dinov2_head | livingbox/modern-style-v4 | only one label present; AUC undefined |
| dinov2_head | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |
| dinov2_head | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| dinov2_head | openskyml/midjourney-mini | only one label present; AUC undefined |
| dinov2_head | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| dinov2_head | plasmo/woolitize | only one label present; AUC undefined |
| dinov2_head | prushton/text-inv-myra | only one label present; AUC undefined |
| dinov2_head | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| dinov2_head | segmind/SSD-1B | only one label present; AUC undefined |
| dinov2_head | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| dinov2_head | songhee/rugged-sporty-car | only one label present; AUC undefined |
| dinov2_head | steb6/textual_inversion_cat | only one label present; AUC undefined |
| dinov2_head | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| random | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| random | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| random | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| random | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| random | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| random | Masagin/Deliberate | only one label present; AUC undefined |
| random | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| random | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| random | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| random | Yntec/CultClassic | only one label present; AUC undefined |
| random | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| random | Yntec/LehinaModel | only one label present; AUC undefined |
| random | Yntec/Paragon | only one label present; AUC undefined |
| random | Yntec/StolenDreams | only one label present; AUC undefined |
| random | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| random | Yntec/fennPhoto | only one label present; AUC undefined |
| random | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| random | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| random | briannlongzhao/0 | only one label present; AUC undefined |
| random | briannlongzhao/2 | only one label present; AUC undefined |
| random | digiplay/Realisian_v1 | only one label present; AUC undefined |
| random | digiplay/Remedy | only one label present; AUC undefined |
| random | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| random | employelalisa/min | only one label present; AUC undefined |
| random | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| random | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| random | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| random | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| random | lewdryuna/A-Rainier | only one label present; AUC undefined |
| random | livingbox/modern-style-v4 | only one label present; AUC undefined |
| random | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| random | none | only one label present; AUC undefined |
| random | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| random | openskyml/midjourney-mini | only one label present; AUC undefined |
| random | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| random | plasmo/woolitize | only one label present; AUC undefined |
| random | prushton/text-inv-myra | only one label present; AUC undefined |
| random | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| random | segmind/SSD-1B | only one label present; AUC undefined |
| random | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| random | songhee/rugged-sporty-car | only one label present; AUC undefined |
| random | steb6/textual_inversion_cat | only one label present; AUC undefined |
| random | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| signals_mean | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| signals_mean | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| signals_mean | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| signals_mean | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| signals_mean | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| signals_mean | Masagin/Deliberate | only one label present; AUC undefined |
| signals_mean | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| signals_mean | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| signals_mean | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| signals_mean | Yntec/CultClassic | only one label present; AUC undefined |
| signals_mean | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| signals_mean | Yntec/LehinaModel | only one label present; AUC undefined |
| signals_mean | Yntec/Paragon | only one label present; AUC undefined |
| signals_mean | Yntec/StolenDreams | only one label present; AUC undefined |
| signals_mean | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| signals_mean | Yntec/fennPhoto | only one label present; AUC undefined |
| signals_mean | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| signals_mean | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| signals_mean | briannlongzhao/0 | only one label present; AUC undefined |
| signals_mean | briannlongzhao/2 | only one label present; AUC undefined |
| signals_mean | digiplay/Realisian_v1 | only one label present; AUC undefined |
| signals_mean | digiplay/Remedy | only one label present; AUC undefined |
| signals_mean | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| signals_mean | employelalisa/min | only one label present; AUC undefined |
| signals_mean | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| signals_mean | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| signals_mean | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| signals_mean | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| signals_mean | lewdryuna/A-Rainier | only one label present; AUC undefined |
| signals_mean | livingbox/modern-style-v4 | only one label present; AUC undefined |
| signals_mean | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| signals_mean | none | only one label present; AUC undefined |
| signals_mean | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| signals_mean | openskyml/midjourney-mini | only one label present; AUC undefined |
| signals_mean | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| signals_mean | plasmo/woolitize | only one label present; AUC undefined |
| signals_mean | prushton/text-inv-myra | only one label present; AUC undefined |
| signals_mean | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| signals_mean | segmind/SSD-1B | only one label present; AUC undefined |
| signals_mean | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| signals_mean | songhee/rugged-sporty-car | only one label present; AUC undefined |
| signals_mean | steb6/textual_inversion_cat | only one label present; AUC undefined |
| signals_mean | suhoRang/textual_inversion_cat | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| constant_fake | Community Forensics | 0.500 |
| constant_real | Community Forensics | 0.500 |
| dinov2_head | Community Forensics | 1.000 |
| random | Community Forensics | 0.492 |
| signals_mean | Community Forensics | 0.341 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| constant_fake | 15000 | 0.01 |
| constant_real | 15000 | 0.01 |
| dinov2_head | 15000 | 43.21 |
| random | 15000 | 0.57 |
| signals_mean | 15000 | 271.93 |
