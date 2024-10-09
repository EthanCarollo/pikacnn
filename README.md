# Tensorflow Pokemon CNN

This is a little repo of a Pokemon CNN that aims to recognize every pokemon of the first generation, using different languages and tools.

> DISCLAIMER : In this repository, you will have some examples of how to make a CNN with different technologies, however, I do not claim to know how to use it at all, like, I don't know anything about the Elixir ecosystem for example, the main purpose of making this CNN in different languages with differents framework was simply for learning purposes and not for real use (even if it may should work lol).

## Structure


```
├── data 
├── elixir
├── gleam
└── jupyter
```

### Data

**Data folder**, it's here where you put the dataset folder with every labels, in, you can also apply data augmentations that are in the jupyter folder.

### Elixir

**Elixir Folder**, a little example of the implementation of the CNN but with [Axon](https://github.com/elixir-nx/axon) (a machine learning library in Elixir)

### Gleam

**Gleam Folder**, a little example of how we can make a little CNN with Tensorflow JS and Gleam, nothing really solid but it works !

### Jupyter

**Jupyter Folder**, a real in world example with Tensorflow of the implementation of a little CNN.


## Datasets 

We can use a large amount of dataset, in our case, we use : 
https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files/data
and then, we just put it into data/pokemon
