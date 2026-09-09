# Benchmark report: Community Forensics

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.124 | 0.250 |
| c2pa | tuned | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.124 | 0.250 |
| copy_move | fixed | 0.503 | 0.379 | 0.626 | 0.503 | 0.000 | 0.005 | 0.075 | 0.240 |
| copy_move | tuned | 0.503 | 0.379 | 0.626 | 0.503 | 0.000 | 0.005 | 0.075 | 0.240 |
| double_jpeg | fixed | 0.497 | 0.375 | 0.624 | 0.500 | 0.000 | 0.000 | 0.076 | 0.240 |
| double_jpeg | tuned | 0.497 | 0.375 | 0.624 | 0.500 | 0.000 | 0.000 | 0.076 | 0.240 |
| ela | fixed | 0.304 | 0.289 | 0.324 | 0.429 | 0.994 | 0.851 | 0.349 | 0.323 |
| ela | tuned | 0.304 | 0.289 | 0.624 | 0.500 | 0.000 | 0.000 | 0.349 | 0.323 |
| jpeg_ghost | fixed | 0.380 | 0.366 | 0.366 | 0.477 | 0.970 | 0.923 | 0.324 | 0.339 |
| jpeg_ghost | tuned | 0.380 | 0.366 | 0.628 | 0.521 | 0.048 | 0.090 | 0.324 | 0.339 |
| metadata | fixed | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.124 | 0.250 |
| metadata | tuned | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.124 | 0.250 |
| sd_watermark | fixed | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.074 | 0.240 |
| sd_watermark | tuned | 0.500 | 0.376 | 0.624 | 0.500 | 0.000 | 0.000 | 0.074 | 0.240 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| c2pa | clean | 0.500 | 0.500 |
| copy_move | clean | 0.503 | 0.503 |
| double_jpeg | clean | 0.497 | 0.500 |
| ela | clean | 0.304 | 0.429 |
| jpeg_ghost | clean | 0.380 | 0.477 |
| metadata | clean | 0.500 | 0.500 |
| sd_watermark | clean | 0.500 | 0.500 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| c2pa | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| c2pa | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| c2pa | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| c2pa | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| c2pa | Masagin/Deliberate | only one label present; AUC undefined |
| c2pa | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| c2pa | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| c2pa | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| c2pa | Yntec/CultClassic | only one label present; AUC undefined |
| c2pa | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| c2pa | Yntec/LehinaModel | only one label present; AUC undefined |
| c2pa | Yntec/Paragon | only one label present; AUC undefined |
| c2pa | Yntec/StolenDreams | only one label present; AUC undefined |
| c2pa | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| c2pa | Yntec/fennPhoto | only one label present; AUC undefined |
| c2pa | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| c2pa | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| c2pa | briannlongzhao/0 | only one label present; AUC undefined |
| c2pa | briannlongzhao/2 | only one label present; AUC undefined |
| c2pa | digiplay/Realisian_v1 | only one label present; AUC undefined |
| c2pa | digiplay/Remedy | only one label present; AUC undefined |
| c2pa | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| c2pa | employelalisa/min | only one label present; AUC undefined |
| c2pa | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| c2pa | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| c2pa | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| c2pa | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| c2pa | lewdryuna/A-Rainier | only one label present; AUC undefined |
| c2pa | livingbox/modern-style-v4 | only one label present; AUC undefined |
| c2pa | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| c2pa | none | only one label present; AUC undefined |
| c2pa | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| c2pa | openskyml/midjourney-mini | only one label present; AUC undefined |
| c2pa | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| c2pa | plasmo/woolitize | only one label present; AUC undefined |
| c2pa | prushton/text-inv-myra | only one label present; AUC undefined |
| c2pa | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| c2pa | segmind/SSD-1B | only one label present; AUC undefined |
| c2pa | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| c2pa | songhee/rugged-sporty-car | only one label present; AUC undefined |
| c2pa | steb6/textual_inversion_cat | only one label present; AUC undefined |
| c2pa | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| copy_move | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| copy_move | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| copy_move | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| copy_move | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| copy_move | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| copy_move | Masagin/Deliberate | only one label present; AUC undefined |
| copy_move | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| copy_move | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| copy_move | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| copy_move | Yntec/CultClassic | only one label present; AUC undefined |
| copy_move | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| copy_move | Yntec/LehinaModel | only one label present; AUC undefined |
| copy_move | Yntec/Paragon | only one label present; AUC undefined |
| copy_move | Yntec/StolenDreams | only one label present; AUC undefined |
| copy_move | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| copy_move | Yntec/fennPhoto | only one label present; AUC undefined |
| copy_move | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| copy_move | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| copy_move | briannlongzhao/0 | only one label present; AUC undefined |
| copy_move | briannlongzhao/2 | only one label present; AUC undefined |
| copy_move | digiplay/Realisian_v1 | only one label present; AUC undefined |
| copy_move | digiplay/Remedy | only one label present; AUC undefined |
| copy_move | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| copy_move | employelalisa/min | only one label present; AUC undefined |
| copy_move | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| copy_move | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| copy_move | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| copy_move | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| copy_move | lewdryuna/A-Rainier | only one label present; AUC undefined |
| copy_move | livingbox/modern-style-v4 | only one label present; AUC undefined |
| copy_move | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| copy_move | none | only one label present; AUC undefined |
| copy_move | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| copy_move | openskyml/midjourney-mini | only one label present; AUC undefined |
| copy_move | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| copy_move | plasmo/woolitize | only one label present; AUC undefined |
| copy_move | prushton/text-inv-myra | only one label present; AUC undefined |
| copy_move | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| copy_move | segmind/SSD-1B | only one label present; AUC undefined |
| copy_move | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| copy_move | songhee/rugged-sporty-car | only one label present; AUC undefined |
| copy_move | steb6/textual_inversion_cat | only one label present; AUC undefined |
| copy_move | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| double_jpeg | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| double_jpeg | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| double_jpeg | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| double_jpeg | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| double_jpeg | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| double_jpeg | Masagin/Deliberate | only one label present; AUC undefined |
| double_jpeg | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| double_jpeg | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| double_jpeg | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| double_jpeg | Yntec/CultClassic | only one label present; AUC undefined |
| double_jpeg | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| double_jpeg | Yntec/LehinaModel | only one label present; AUC undefined |
| double_jpeg | Yntec/Paragon | only one label present; AUC undefined |
| double_jpeg | Yntec/StolenDreams | only one label present; AUC undefined |
| double_jpeg | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| double_jpeg | Yntec/fennPhoto | only one label present; AUC undefined |
| double_jpeg | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| double_jpeg | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| double_jpeg | briannlongzhao/0 | only one label present; AUC undefined |
| double_jpeg | briannlongzhao/2 | only one label present; AUC undefined |
| double_jpeg | digiplay/Realisian_v1 | only one label present; AUC undefined |
| double_jpeg | digiplay/Remedy | only one label present; AUC undefined |
| double_jpeg | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| double_jpeg | employelalisa/min | only one label present; AUC undefined |
| double_jpeg | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| double_jpeg | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| double_jpeg | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| double_jpeg | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| double_jpeg | lewdryuna/A-Rainier | only one label present; AUC undefined |
| double_jpeg | livingbox/modern-style-v4 | only one label present; AUC undefined |
| double_jpeg | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| double_jpeg | none | only one label present; AUC undefined |
| double_jpeg | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| double_jpeg | openskyml/midjourney-mini | only one label present; AUC undefined |
| double_jpeg | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| double_jpeg | plasmo/woolitize | only one label present; AUC undefined |
| double_jpeg | prushton/text-inv-myra | only one label present; AUC undefined |
| double_jpeg | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| double_jpeg | segmind/SSD-1B | only one label present; AUC undefined |
| double_jpeg | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| double_jpeg | songhee/rugged-sporty-car | only one label present; AUC undefined |
| double_jpeg | steb6/textual_inversion_cat | only one label present; AUC undefined |
| double_jpeg | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| ela | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| ela | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| ela | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| ela | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| ela | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| ela | Masagin/Deliberate | only one label present; AUC undefined |
| ela | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| ela | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| ela | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| ela | Yntec/CultClassic | only one label present; AUC undefined |
| ela | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| ela | Yntec/LehinaModel | only one label present; AUC undefined |
| ela | Yntec/Paragon | only one label present; AUC undefined |
| ela | Yntec/StolenDreams | only one label present; AUC undefined |
| ela | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| ela | Yntec/fennPhoto | only one label present; AUC undefined |
| ela | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| ela | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| ela | briannlongzhao/0 | only one label present; AUC undefined |
| ela | briannlongzhao/2 | only one label present; AUC undefined |
| ela | digiplay/Realisian_v1 | only one label present; AUC undefined |
| ela | digiplay/Remedy | only one label present; AUC undefined |
| ela | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| ela | employelalisa/min | only one label present; AUC undefined |
| ela | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| ela | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| ela | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| ela | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| ela | lewdryuna/A-Rainier | only one label present; AUC undefined |
| ela | livingbox/modern-style-v4 | only one label present; AUC undefined |
| ela | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| ela | none | only one label present; AUC undefined |
| ela | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| ela | openskyml/midjourney-mini | only one label present; AUC undefined |
| ela | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| ela | plasmo/woolitize | only one label present; AUC undefined |
| ela | prushton/text-inv-myra | only one label present; AUC undefined |
| ela | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| ela | segmind/SSD-1B | only one label present; AUC undefined |
| ela | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| ela | songhee/rugged-sporty-car | only one label present; AUC undefined |
| ela | steb6/textual_inversion_cat | only one label present; AUC undefined |
| ela | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| jpeg_ghost | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| jpeg_ghost | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| jpeg_ghost | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| jpeg_ghost | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| jpeg_ghost | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| jpeg_ghost | Masagin/Deliberate | only one label present; AUC undefined |
| jpeg_ghost | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| jpeg_ghost | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| jpeg_ghost | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| jpeg_ghost | Yntec/CultClassic | only one label present; AUC undefined |
| jpeg_ghost | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| jpeg_ghost | Yntec/LehinaModel | only one label present; AUC undefined |
| jpeg_ghost | Yntec/Paragon | only one label present; AUC undefined |
| jpeg_ghost | Yntec/StolenDreams | only one label present; AUC undefined |
| jpeg_ghost | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| jpeg_ghost | Yntec/fennPhoto | only one label present; AUC undefined |
| jpeg_ghost | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| jpeg_ghost | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| jpeg_ghost | briannlongzhao/0 | only one label present; AUC undefined |
| jpeg_ghost | briannlongzhao/2 | only one label present; AUC undefined |
| jpeg_ghost | digiplay/Realisian_v1 | only one label present; AUC undefined |
| jpeg_ghost | digiplay/Remedy | only one label present; AUC undefined |
| jpeg_ghost | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| jpeg_ghost | employelalisa/min | only one label present; AUC undefined |
| jpeg_ghost | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| jpeg_ghost | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| jpeg_ghost | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| jpeg_ghost | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| jpeg_ghost | lewdryuna/A-Rainier | only one label present; AUC undefined |
| jpeg_ghost | livingbox/modern-style-v4 | only one label present; AUC undefined |
| jpeg_ghost | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| jpeg_ghost | none | only one label present; AUC undefined |
| jpeg_ghost | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| jpeg_ghost | openskyml/midjourney-mini | only one label present; AUC undefined |
| jpeg_ghost | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| jpeg_ghost | plasmo/woolitize | only one label present; AUC undefined |
| jpeg_ghost | prushton/text-inv-myra | only one label present; AUC undefined |
| jpeg_ghost | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| jpeg_ghost | segmind/SSD-1B | only one label present; AUC undefined |
| jpeg_ghost | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| jpeg_ghost | songhee/rugged-sporty-car | only one label present; AUC undefined |
| jpeg_ghost | steb6/textual_inversion_cat | only one label present; AUC undefined |
| jpeg_ghost | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| metadata | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| metadata | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| metadata | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| metadata | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| metadata | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| metadata | Masagin/Deliberate | only one label present; AUC undefined |
| metadata | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| metadata | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| metadata | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| metadata | Yntec/CultClassic | only one label present; AUC undefined |
| metadata | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| metadata | Yntec/LehinaModel | only one label present; AUC undefined |
| metadata | Yntec/Paragon | only one label present; AUC undefined |
| metadata | Yntec/StolenDreams | only one label present; AUC undefined |
| metadata | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| metadata | Yntec/fennPhoto | only one label present; AUC undefined |
| metadata | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| metadata | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| metadata | briannlongzhao/0 | only one label present; AUC undefined |
| metadata | briannlongzhao/2 | only one label present; AUC undefined |
| metadata | digiplay/Realisian_v1 | only one label present; AUC undefined |
| metadata | digiplay/Remedy | only one label present; AUC undefined |
| metadata | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| metadata | employelalisa/min | only one label present; AUC undefined |
| metadata | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| metadata | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| metadata | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| metadata | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| metadata | lewdryuna/A-Rainier | only one label present; AUC undefined |
| metadata | livingbox/modern-style-v4 | only one label present; AUC undefined |
| metadata | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| metadata | none | only one label present; AUC undefined |
| metadata | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| metadata | openskyml/midjourney-mini | only one label present; AUC undefined |
| metadata | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| metadata | plasmo/woolitize | only one label present; AUC undefined |
| metadata | prushton/text-inv-myra | only one label present; AUC undefined |
| metadata | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| metadata | segmind/SSD-1B | only one label present; AUC undefined |
| metadata | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| metadata | songhee/rugged-sporty-car | only one label present; AUC undefined |
| metadata | steb6/textual_inversion_cat | only one label present; AUC undefined |
| metadata | suhoRang/textual_inversion_cat | only one label present; AUC undefined |
| sd_watermark | AACEE/textual_inversion_cat | only one label present; AUC undefined |
| sd_watermark | Fictiverse/Stable_Diffusion_BalloonArt_Model | only one label present; AUC undefined |
| sd_watermark | Fictiverse/Stable_Diffusion_PaperCut_Model | only one label present; AUC undefined |
| sd_watermark | Fictiverse/Stable_Diffusion_VoxelArt_Model | only one label present; AUC undefined |
| sd_watermark | ItsJayQz/Marvel_WhatIf_Diffusion | only one label present; AUC undefined |
| sd_watermark | Masagin/Deliberate | only one label present; AUC undefined |
| sd_watermark | Subhamoy12/my-pet-cat-xzr | only one label present; AUC undefined |
| sd_watermark | Sygil/Sygil-Diffusion | only one label present; AUC undefined |
| sd_watermark | WarriorMama777/AbyssOrangeMix | only one label present; AUC undefined |
| sd_watermark | Yntec/CultClassic | only one label present; AUC undefined |
| sd_watermark | Yntec/DreamLikeRemix | only one label present; AUC undefined |
| sd_watermark | Yntec/LehinaModel | only one label present; AUC undefined |
| sd_watermark | Yntec/Paragon | only one label present; AUC undefined |
| sd_watermark | Yntec/StolenDreams | only one label present; AUC undefined |
| sd_watermark | Yntec/epiCRealismVAE | only one label present; AUC undefined |
| sd_watermark | Yntec/fennPhoto | only one label present; AUC undefined |
| sd_watermark | artificialguybr/IconsMI-AppIconsModelforSD | only one label present; AUC undefined |
| sd_watermark | botp/Realistic_Vision_V1.4 | only one label present; AUC undefined |
| sd_watermark | briannlongzhao/0 | only one label present; AUC undefined |
| sd_watermark | briannlongzhao/2 | only one label present; AUC undefined |
| sd_watermark | digiplay/Realisian_v1 | only one label present; AUC undefined |
| sd_watermark | digiplay/Remedy | only one label present; AUC undefined |
| sd_watermark | digiplay/majicMIX_realistic_v1 | only one label present; AUC undefined |
| sd_watermark | employelalisa/min | only one label present; AUC undefined |
| sd_watermark | giangvlcs/LongGiang_textual_inversion_face_v2 | only one label present; AUC undefined |
| sd_watermark | giangvlcs/textual_inversion_cat | only one label present; AUC undefined |
| sd_watermark | iamgokul/my-pet-dog-ggs | only one label present; AUC undefined |
| sd_watermark | kalebanana/textual_inversion_mvtec | only one label present; AUC undefined |
| sd_watermark | lewdryuna/A-Rainier | only one label present; AUC undefined |
| sd_watermark | livingbox/modern-style-v4 | only one label present; AUC undefined |
| sd_watermark | naclbit/trinart_stable_diffusion_v2 | only one label present; AUC undefined |
| sd_watermark | none | only one label present; AUC undefined |
| sd_watermark | nota-ai/bk-sdm-small | only one label present; AUC undefined |
| sd_watermark | openskyml/midjourney-mini | only one label present; AUC undefined |
| sd_watermark | openskyml/open-diffusion-v1 | only one label present; AUC undefined |
| sd_watermark | plasmo/woolitize | only one label present; AUC undefined |
| sd_watermark | prushton/text-inv-myra | only one label present; AUC undefined |
| sd_watermark | prushton/text-inv-myra_fridec8 | only one label present; AUC undefined |
| sd_watermark | segmind/SSD-1B | only one label present; AUC undefined |
| sd_watermark | soltustik/textual_inversion_cat | only one label present; AUC undefined |
| sd_watermark | songhee/rugged-sporty-car | only one label present; AUC undefined |
| sd_watermark | steb6/textual_inversion_cat | only one label present; AUC undefined |
| sd_watermark | suhoRang/textual_inversion_cat | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| c2pa | Community Forensics | 0.500 |
| copy_move | Community Forensics | 0.503 |
| double_jpeg | Community Forensics | 0.497 |
| ela | Community Forensics | 0.304 |
| jpeg_ghost | Community Forensics | 0.380 |
| metadata | Community Forensics | 0.500 |
| sd_watermark | Community Forensics | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 1000 | 0.38 |
| copy_move | 1000 | 61.72 |
| double_jpeg | 1000 | 5.24 |
| ela | 1000 | 22.18 |
| jpeg_ghost | 1000 | 118.01 |
| metadata | 1000 | 7.47 |
| sd_watermark | 1000 | 52.37 |
