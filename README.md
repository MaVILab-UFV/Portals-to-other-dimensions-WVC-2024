# Portal to Other Dimensions: use of Computer Vision to create art work from day life images

This repository contains the proposed method of the paper **[Portal to Other Dimensions: use of Computer Vision to create art work from day life images]()** published at *Workshop on Computer Vision (WVC), 2024*.

We propose a method to use Computer Vision to create art from our daily lives as a way to demonstrate its applications. To achieve this, we apply object segmentation to detect a door in each frame and use style transfer networks to stylize the input frame. Then, we employ digital image processing techniques to stylize the interior of the door, giving the impression that it is a portal to another dimension.

If you find this code useful for your research, please cite the paper:


## Getting Started

You can install the project dependencies using conda, we provide an `environment.yml` file with the frozen dependencies. First, create an environment named `portal` with all the necessary dependencies:

```bash
conda env create -f environment.yml -n portal
```

Then, activate the environment:

```bash
conda activate portal
```

If you want to use the pretrained network with 16 styles, which requires less GPU memory, you need to download the pretrained model [here](https://github.com/ryanwongsa/Real-time-multi-style-transfer). Also, you can install the project dependencies using pip, we provide an  `requirements.txt` file with the frozen dependencies:

```bash
pip install -r requirements
```

## Real Time Application To Convert Doors into Portals
<details>
<summary>Using Pretrained Network with 16 Styles</summary>
Execute `realtime.py` script

```bash
python realtime.py --save-video --style $style_number --model_path $model_path
```

If you want to save the video after the program ended, you add the flag 'save-video'.
</details>


<details>
<summary>Using Arbitrary Transfer Style</summary>
Execute `realtime_arbitrary.py` script

```bash
python realtime.py --save-video --style_path $style_path 
```

If you want to save the video after the program ended, you add the flag 'save-video'.
</details>











## Contact
### Authors
---


| [Luísa Ferreira](https://github.com/ferreiraluisa) | [Ana Osias](https://github.com/AnaClaraOsias) | [Lucas Vieira]() |[Michel Silva](https://michelmelosilva.github.io/) |
| :--------------------------------------------: | :------------------------------------------------: | :------------------------------------------------:  |:------------------------------------------------: |
|                 BSc. Student¹                  |                   MSc. Student¹                    |                BSc. Student¹                |      Assistant Professor¹                |
|          <luisa.ferreira@ufv.br>           |              <ana.osias@ufv.br>               |             <lucas.v.santos@ufv.br>               |              <michel.m.silva@ufv.br>               |

¹Universidade Federal de Viçosa \
Departamento de Informática \
Viçosa, Minas Gerais, Brazil


---
### Laboratory

![MaVILab](https://mavilab-ufv.github.io/images/mavilab-logo.png) | ![UFV](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStNoEd-f21Ji2O518vY3ox0AaEK38uKiiJYg&s)
--- | ---


**MaVILab:** Machine Vision and Intelligence Laboratory
https://mavilab-ufv.github.io/




### Enjoy it! :smiley: