# Benchmark report: Community Forensics+WildRF

- Entries evaluated: 5564
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 1.000 | 0.999 | 0.994 | 0.993 | 0.005 | 0.991 | 0.005 | 0.006 |
| dinov2_head | tuned | 1.000 | 0.999 | 0.990 | 0.992 | 0.015 | 0.999 | 0.005 | 0.006 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 1.000 | 0.993 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
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

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | COCO | only one label present; AUC undefined |
| dinov2_head | Community Forensics | 1.000 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 5564 | 52.89 |
