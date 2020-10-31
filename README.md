# Machine translation

## Conda environement

A 'yml' file containing a conda environment can be used to generate a proper environment in order to launch this project

[Conda cheat sheet]: https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf

TODO : Some library aren't referenced by conda. Write the list of dependencies in a .txt and ask to install them through pip



## Parrallel datasets

The datasets have to be decompressed and placed in the **datasets** folder

[Wikipedia en-de](http://opus.nlpl.eu/download.php?f=Wikipedia%2Fv1.0%2Ftmx%2Fde-en.tmx.gz )

[Additional references to // datasets](https://lionbridge.ai/datasets/25-best-parallel-text-datasets-for-machine-translation-training/ )

[Additional references to // datasets](http://opus.nlpl.eu/ )

## Pre Trained embeddings vectors

The embeddings files containing vectors have to be decompressed and placed in the **embeddings** folder

[google word2vec en](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

[glove en](http://nlp.stanford.edu/data/glove.840B.300d.zip)

[fasttext en](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz)

[fasttext de](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz)

```bash
curl -o de_glove.txt https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt
curl -o de_word2vec.txt https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt
```

## Notes

If there is a problem in calling the file, check *filespath.py* and see if the names of the directories and files are correct

