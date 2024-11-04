# Portal to Other Dimensions: use of Computer Vision to create art work from day life images

This repository contains the proposed method of the paper **[Portal to Other Dimensions: use of Computer Vision to create art work from day life images]()** published at *Workshop on Computer Vision (WVC), 2024*.

We propose a method to use Computer Vision to create art from our daily lives as a way to demonstrate its applications. To achieve this, we apply object segmentation to detect a door in each frame and use style transfer networks to stylize the input frame. Then, we employ digital image processing techniques to stylize the interior of the door, giving the impression that it is a portal to another dimension.

If you find this code useful for your research, please cite the paper:

```bibtex
@inproceedings{silva2024,
    title        = {Portal to Other Dimensions: use of Computer Vision to create art work from day life images},
    author       = {Ferreira, Luísa and Osias, Ana and Vieira, Lucas and Silva, Michel},
    year         = 2024,
    booktitle    = {Anais do XIX Workshop de Visão Computacional}, 
    location     = {Rio Paranaíba/MG},
    address      = {Rio Paranaíba, MG, Brasil}
    publisher    = {SBC}, 
    doi          = {}, 
    url          = {}, 
    note         = {to appear}
}
```


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

Please ensure that the file `best.pt` is included in your project, as it contains the weights for the fine-tuned YOLO model used for classifying and segmenting doors.

## Real Time Application To Convert Doors into Portals
<details>
<summary>Using Pretrained Network with 16 Styles</summary>
Execute `realtime.py` script

```bash
python realtime.py --save-video --style <style_number> --model_path <path_to_model>
```

<ul>
  <li><code>--save-video</code>: This optional flag saves the video when the program runs.</li>
  <li><code>--style</code>: Set <code>&lt;style_number&gt;</code> to a number from 0 to 15, representing each of the 16 styles the network is trained to transfer.</li>
  <li><code>--model_path</code>: Specify <code>&lt;path_to_model&gt;</code> as the path to the pre-trained style transfer model, which can be downloaded from <a href="https://github.com/ryanwongsa/Real-time-multi-style-transfer">here</a>.</li>
</ul>

</details>


<details>
<summary>Using Arbitrary Transfer Style</summary>

Execute `realtime_arbitrary.py` script. The network to transfer style from arbitrary image needs more GPU memory to work(>= 3GB VRAM).

```bash
python realtime_arbitrary.py --save-video --style_path <path_to_style_image>
```
<ul>
  <li><code>--save-video</code>: This optional flag saves the video when the program runs.</li>
  <li> <code>--style_path</code>: Replace <code>&lt;path_to_style_image&gt;</code> with the path to the image you want to use as the style.</li>
</ul>

</details>



## Acknowledgments 

We would like to thank [@ryanwongsa](https://github.com/ryanwongsa) for making the model and [code](https://github.com/ryanwongsa/Real-time-multi-style-transfer?tab=readme-ov-file) for his implementation of Real-Time Style Transfer using PyTorch with 16 styles available. We also extend our gratitude to [Ghiasi et al.](https://arxiv.org/abs/1705.06830) for providing the model for arbitrary neural artistic stylization.


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
