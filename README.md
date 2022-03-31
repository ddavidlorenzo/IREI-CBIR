# Implementation of a basic CBIR

### Lorenzo Alfaro, David

### Morote García, Carlos

#### ETSIINF, UPM

---

## Motivation

Content-based image retrieval (CBIR) aims at finding similar images from large scale datasets against query images. The degree of commonality between images is often computed accounting for (i) a set of representative features of each sample, and (ii) a set of similarity measures, e.g., distance functions, that quantify the extent to which each feature in each pair of samples matches. Visual cues such as colour, texture and shape are among the most prominent image feature descriptors, and exploiting the appropriate ones for a given problem is key to the success of the retrieval process.

In this work, we review various techniques that enable users retrieve images, provided a set of meaningful search criteria. We state such criteria to be meaningful in the sense that they are easy and fast to formulate, i.e., effort and time required for the user to specify search conditions are measurably lesser than manually inspecting the dataset; and in such manner that they help to best describe and discriminate target images.

Specifically, data under consideration, published in (Kolman, 2021), is a collection of images of 143 different Pokemons, fictional creatures characterized by their inter-class heterogeneous colours and shapes. Regardless of the very problem domain, what we hereby aim at reviewing is a set of simple techniques that may be transversely applicable to other paradigms of interest where those descriptors are of vital relevance. Namely, we explore the use of different histogram comparison methods, unsupervised classification techniques to yield the most relevant colours of an image, and algorithms for Keypoint description such as SIFT (Lowe, 1999).

![](https://irei-cbir.readthedocs.io/en/latest/_images/i1.png)

---

## Run the code

To run the code, once the dependencies has been installed, it must be done through the file `CBIR.py` like:

`python CBIR.py <args>`

In order to know the parameters that the function uses type:

`python CBIR.py -h`

---

### Execution examples

**Colour search**

`python CBIR.py "pokemon_dataset" col -c custom -t 15 -se`

**Histogram search**
`python CBIR.py "pokemon_dataset" hist -i "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg" -t 15 -se -cm bhatta`

**Smart histogram search**
`python CBIR.py "pokemon_dataset" smarthist -i "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg" -t 15 -hg 5 -wg 5 -se -cm bhatta`

---

## Code dependencies

 - matplotlib
 - numpy
 - opencv_python
 - pickle

---

## Contents of the directory

```
.
.
├── README.md
├── CBIR.py
├── SIFT.py
├── colour_search.py
├── docs
│   └── ...
├── hist_search.py
├── img_search.py
├── pokemon_dataset
│   └── ...
├── requirements.txt
├── serial
│   ├── colour_serial.pkl
│   └── hist_serial.pkl
├── test.py
├── tree.txt
└── utils.py
```

---