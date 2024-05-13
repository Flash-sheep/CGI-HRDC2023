# VSEE: Vision Semantic Evaluation for Hypertensive Retinopathy

## Environment

* Install in your environment a compatible torch version with your GPU. For example:
```
conda create -n vsee python=3.8 -y
conda activate vsee
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```


## Usage

```
from PIL import Image
import numpy as np

from vsee import FLAIRModel

model = FLAIRModel(from_checkpoint=True)

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)

image = np.array(Image.open("./documents/sample_macular_hole.png"))
text = ["normal", "healthy", "macular edema", "diabetic retinopathy", "glaucoma", "macular hole",
        "lesion", "lesion in the macula"]

# Forward FLAIR model to compute similarities
probs, logits = model(image, text)
```

## Pretrained model (if you use it, you can skip pre-training step)


| Backbone  |      ID      |                                                                                               |
|-----------|:------------:|:---------------------------------------------------------------------------------------------:|
| ResNet-50 | flair_resnet | [LINK](https://drive.google.com/file/d/1l24_2IzwQdnaa034I0zcyDLs_zMujsbR/view?usp=drive_link) |

## Pre-training and transferability

In the following, we present the scripts for model pre-training and transferability. To use them, we recommend cloning the whole repository.

```
git clone https://github.com/Flash-sheep/CGI-HRDC2023.git
cd CGI-HRDC2023
pip install -r requirements.txt
```

### 📦 Foundation model pre-training

* Prepare the FUNDUS assembly dataset - check `./local_data/prepare_partitions.py` to prepare the dataframes.


|                                                                                                                                       |                                                                                                                           |                                                                                        |                                                                             |                                                                                                                                                                 |                                                                                                         |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [01_EYEPACS](https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess)                                                         | [08_ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)                                 | [15_APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)     | [22_HEI-MED](https://github.com/lgiancaUTH/HEI-MED)                         | [29_AIROGS](https://zenodo.org/record/5793241#.ZDi2vNLMJH5)                                                                                                     | [36_ACRIMA](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y) |
| [02_MESIDOR](https://www.adcis.net/en/third-party/messidor2/)                                                                         | [09_PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004/1)                                                      | [16_FUND-OCT](https://data.mendeley.com/datasets/trghs22fpg/3)                         | [23_HRF](http://www5.cs.fau.de/research/data/fundus-images/)                | [30_SUSTech-SYSU](https://figshare.com/articles/dataset/The_SUSTech-SYSU_dataset_for_automated_exudate_detection_and_diabetic_retinopathy_grading/12570770/1)   | [37_DeepDRiD](https://github.com/deepdrdoc/DeepDRiD)                                                    |
| [03_IDRID](https://idrid.grand-challenge.org/Rules/)                                                                                  | [10_PARAGUAY](https://zenodo.org/record/4647952#.ZBT5xXbMJD9)                                                             | [17_DiaRetDB1](https://www.it.lut.fi/project/imageret/diaretdb1_v2_1/)                 | [24_ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/)                       | [31_JICHI](https://figshare.com/articles/figure/Davis_Grading_of_One_and_Concatenated_Figures/4879853/1)                                                        |                                                                                                         |
| [04_RFMid](https://ieee-dataport.org/documents/retinal-fundus-multi-disease-image-dataset-rfmid-20)                                   | [11_STARE](https://cecas.clemson.edu/~ahoover/stare/)                                                                     | [18_DRIONS-DB](http://www.ia.uned.es/~ejcarmona/DRIONS-DB.html)                        | [25_REFUGE](https://refuge.grand-challenge.org/)                            | [32_CHAKSU](https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135?file=38944805)                                      |                                                                                                         |
| [05_1000x39](https://www.nature.com/articles/s41467-021-25138-w#Sec16)                                                                | [12_ARIA](https://www.damianjjfarnell.com/?page_id=276)                                                                   | [19_Drishti-GS1](http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | [26_ROC](http://webeye.ophth.uiowa.edu/ROC/)                                | [33_DR1-2](https://figshare.com/articles/dataset/Advancing_Bag_of_Visual_Words_Representations_for_Lesion_Classification_in_Retinal_Images/953671?file=6502302) |                                                                                                         |
| [06_DEN](https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation)  | [13_FIVES](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1) | [20_E-ophta](https://www.adcis.net/en/third-party/e-ophtha/)                           | [27_BRSET](https://physionet.org/content/brazilian-ophthalmological/1.0.0/) | [34_Cataract](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)                                                                                           |                                                                                                         |
| [07_LAG](https://github.com/smilell/AG-CNN)                                                                                           | [14_AGAR300](https://ieee-dataport.org/open-access/diabetic-retinopathy-fundus-image-datasetagar300)                      | [21_G1020](https://arxiv.org/abs/2006.09158)                                           | [28_OIA-DDR](https://github.com/nkicsl/DDR-dataset)                         | [35_ScarDat](https://github.com/li-xirong/fundus10k)                                                                                                            |                                                                                                         |


* Vision-Language Pre-training.

```
python main_pretrain.py --augment_description True --balance True --epochs 15 --batch_size 128 --num_workers 6
```

### 📦 Transferability to downstream tasks/domains
* Zero-shot (no adaptation).

```
python main_transferability.py --experiment CGI_HRDC_Task2 --method zero_shot --load_weights True --domain_knowledge True  --shots_train 0% --shots_test 100% --project_features True --norm_features True --folds 1 
```

* Linear Probing.

```
python main_transferability.py --experiment CGI_HRDC_Task2 --method lp --load_weights True --shots_train 80% --shots_test 20% --project_features False --norm_features False --folds 5 
```
