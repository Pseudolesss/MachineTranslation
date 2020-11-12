# Machine translation

## Conda environement

A 'yml' file containing a conda environment can be used to generate a proper environment in order to launch this project

```bash
cd project-root/
conda env create --file environment.yml
conda activate machineTranslation36
jupyter notebook
```

## File needed for running the notebook

The required files (prepared dataset, embeddings and trained models) are compressed in an archive at this address : [Needed Files Archive](https://drive.google.com/file/d/1pYErSWHwz6JelbCFruM4Kw2BRYN1jWIW/view?usp=sharing)

Just download it straightaway and decompress it and place the folders at the project root. The following links are just to indicates the sources of the files. If a problem occurs, do not hesitate to notify one of us by email (j.lhoest@student.uliege.be)

### Parallel dataset

The dataset have to be decompressed and placed in the **datasets** folder. The data is presented in a tmx file (translation memory file) and was provided by the Opus project. The dataset used consists in German sentences as sources and English sentences as targets. The corpus consists in indication from medication notices. 

[European Medicines Agency corpus](http://opus.nlpl.eu/ELRC_2682-v1.php )

### Pre Trained embeddings vectors

The embeddings files containing vectors have to be decompressed and placed in the **embeddings** folder

[google word2vec en](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

[glove en](http://nlp.stanford.edu/data/glove.840B.300d.zip)

[fasttext en](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz)

[fasttext de](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz)

[word2vec de & glove de](https://deepset.ai/german-word-embeddings) (see command to download them)

```bash
curl -o DE_glove.txt https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt
curl -o DE_word2vec.txt https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt
```



## Notes

Only the DE word2vec and DE glove embeddings come from other institutions than the one which have design the models (Google and Stanford). It is possible that the results could be invalid because of the validity of the files which is unknowed.

