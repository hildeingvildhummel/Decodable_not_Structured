# Decodable_not_Structured
Implementation of embedding analysis of pretrained audio models applied to ShipRadiatedNoise recognition

# Audio Representation Evaluation Pipeline

This repository provides a **flexible evaluation pipeline for audio representations**, supporting both **supervised** and **unsupervised** evaluation of features extracted from pretrained models (e.g. MelSpectrograms, BEATS, HuBERT, AudioMAE, etc.).

The main entry point is a single script (`main.py`) that allows you to run experiments via the command line in a **reproducible and configurable** way.

---

## ✨ Features

- Multiple audio representation backends (e.g. MelSpec, BEATS, HuBERT, WavLM)
- Supervised evaluation
  - Linear classifier
  - Attentive classifier
- Unsupervised evaluation
  - Clustering with Mutual Information
  - Rank-based similarity metrics
- Clean CLI interface for experiments
- Easy extension to new models or datasets

---

## 📁 Expected Project Structure

```text
project/
├── main.py
├── loaders.py                  # data loaders (assumed)
├── models.py                   # feature extractors (assumed)
├── Supervised_evaluation.py
├── Unsupervised_evaluation.py
├── Similarity.py
├── models/
│   └── Classifiers/
└── Data/
    └── <dataset_name>/
        ├── train/
        └── test/
```
## ⚠️ Assumed is that the embeddings of the bioacoustic models are already extracted using: https://github.com/bioacoustic-ai/bacpipe

## 🚀 Usage
### Basic command (unsupervised only)

python3 main.py
--train_dir /path/to/train
--test_dir /path/to/test
--model MelSpec
--unsupervised

### Supervised evaluation (linear classifier)

python3 main.py
--train_dir /path/to/train
--test_dir /path/to/test
--model BEATS
--classifier linear
--supervised

### Supervised evaluation (attentive classifier)

python3 main.py
--train_dir /path/to/train
--test_dir /path/to/test
--model BEATS
--classifier attentive
--supervised

### Combined supervised + unsupervised evaluation

python3 main.py
--train_dir /path/to/train
--test_dir /path/to/test
--model HuBERT
--supervised
--unsupervised

## ⚙️ Command-Line Arguments
| Argument | Type | Description |
|----------|------|-------------|
| --train_dir | str | Path to training dataset directory (required) |
| --test_dir | str | Path to test dataset directory (required) |
| --model | str | Feature extractor (BEATS, HuBERT, AudioMAE, WavLM, Data2vec, MelSpec) |
| --classifier | str | linear or attentive (supervised only) |
| --model_out | str | Path to save trained classifier |
| --supervised | flag | Enable supervised evaluation |
| --unsupervised | flag | Enable unsupervised evaluation |

## 📊 Outputs

Depending on the selected options, the script produces the following outputs.

### Supervised evaluation
- Classification accuracy
- Confusion matrix
- Optional rank-similarity scores on classifier logits
- Saved classifier model (`.pt` file)

### Unsupervised evaluation
- Clustering Mutual Information (MI) score
- Rank-based similarity scores on embeddings

### Saved files
- Trained classifiers are saved to:

  models/Classifiers/<model_name>.pt

## 🧠 Supported Models

The following feature extractors are supported. The table includes **embedding dimensions** and **links to the original papers**:

| Model      | Embedding Dimension | Supervised | Unsupervised | Paper / Reference |
|-----------|------------------|------------|--------------|-----------------|
| Animal2Vec  | 1024              | ✅ | ✅ | [Animal2Vec](https://arxiv.org/pdf/2406.01253) |
| AudioMAE  | 768              | ✅ | ✅ | [AudioMAE](https://arxiv.org/pdf/2207.06405) |
| AVES  | 768              | ✅ | ✅ | [AVES](https://arxiv.org/pdf/2210.14493) |
| AvesEcho  | 768              | ✅ | ✅ | [AvesEcho](https://arxiv.org/pdf/2409.15383?) |
| BEATS     | 768              | ✅ | ✅ | [BEATS: Audio Representation Learning](https://arxiv.org/pdf/2212.09058) |
| BirdMAE  | 1280              | ✅ | ✅ | [BirdMAE](https://arxiv.org/pdf/2504.12880) |
| BirdNet  | 1024              | ✅ | ✅ | [BirdNet](https://pdf.sciencedirectassets.com/273474/1-s2.0-S1574954120X00069/1-s2.0-S1574954121000273/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHfSXXlu%2FzC%2BXFJmMj%2BqK81AmCj%2FKv503xqCaStmm6MpAiAbg%2BEpXUcPxeBxG2KbVA68gm7%2FrRJDIXhtqamHcQPydyq8BQjF%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMVQdit2b1Zz3KoGgnKpAF0Pu7kigQb1p9LqRf333RMyw6cOUrUTHE%2BehR4BQoP%2FqIgXGE%2B3nsi5oVZQc8xDUA9uxKxUqVvEEZ7bLvx63rbpN2El8pqSQnIyqaCpRbYqkFAk3c1opJoHEowavYtx2phy4XMrglBRvmObDii189A8BvgtaOgCz8YWd5JHPglfayLTVHd4mfkl2uRgbCbeAQ5N2TJtTCl04xA%2BiDM3KmWIuN8Hcai14081f1YT7pXJzUr%2F3a7BZNMYNDIpIWSugJXJB1SYFEoiotgfIv3rTQG4a3lCWDZiyr0g%2FzmClUMrU242pNwX8%2Fr4wJe778Hv2cYDIbDNcVeYzw6ujp%2FEQXUIWpS%2Bn3R21m4wUdvDu13AxIonJi8knp0m7xQM0B0SZZFpx5%2F7%2B0ZafZDY00EqNktTMPtz%2FfHT3AWe%2FUahbg7r8xmrb%2BNj1XqVANuJR6TEhkCjOPulGFQC26WXYoOjNtuwQM4dKaVgyek4A4VUIBzoR08VxV8CVMhuU0YtG%2BYowTez%2FkFO4u8hGcvPYaB0sDhasNP0evHe%2BtaYvUpuHf0h86b%2FOwerQ1TI1ny8is8GtJ4f%2BsD7o9uIGKCIlRMsogqqeBx0OEkMsWyFhb1zbLz5Ea4wL9%2BALH3MRZzK%2BKNoAQWVNYZO2G%2BQ4LeakVfG%2B8XWW0UK1%2FypKTTjCotVm6ZYuEnAXDx1hD2HB0Hl1cvvHwa%2FvsUZxbmDGfuYzPZO2KNidytlxMXcUy3BRXbUbbLFghs8Z9a6EszTytU7XtLHuzxyDws%2FrKV3HN2AGtJJKNvQGBnFvx9sELgwa2LSRtHk2ZWL%2B2C1yr5ssBeLf7VvmQuwkPcst1%2BQIf2lW3tieqrZFfND0Xkb73%2FTpgQuZcso0wv%2FnCywY6sgH8%2FRiPEXne0JTCoMTHj2W8Av2ABwtAZpAiOP5p0YNqD4GiCfae2W1c%2Fgkf3EHC8ikjqgldz9g788%2BLsgNnmczg%2B1yeGzNszyeAolWUY96QBybQSXCRIHni8uMPFN8yIquo8GNwvsonYB%2FYmFmO186pXinqqA00%2FmBUbcEdBwiubOsvHfigr6ecwwInU8DZ7ZEBIHqzgSuW2HgdoP6YnFZ6bDNrWRJum3ybUQ4ALQTROZ9J&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260121T130543Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY443QIMC2%2F20260121%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c4ce3da153ddb5a82b3515277c735e85974da051e07156b4097bf2c45e82f8bd&hash=c96081c7493eabf28efe45eeb89dfcf14676ee5b7f5c0b3f1b7834d0731db613&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1574954121000273&tid=spdf-d6f85e94-7d96-4678-a300-75877989f6e4&sid=9489a415679bb54a1f49b908f07dfa0fc78bgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=000c560455575f50&rr=9c1706550832b5ad&cc=nl) |
| Data2vec  | 768              | ✅ | ✅ | [Data2vec](https://arxiv.org/abs/2202.03555) |
| GoogleWhale  | 1280              | ✅ | ✅ | [GoogleWhale](https://www.kaggle.com/models/google/multispecies-whale) |
| HuBERT    | 768              | ✅ | ✅ | [HuBERT](https://arxiv.org/abs/2106.07447) |
| HuBERTAS  | 768              | ✅ | ✅ | [HuBERTAS](https://dataloop.ai/library/model/alm_hubert-base-audioset/) |
| MelSpec   | 128  | ✅ | ✅ | [Mel Spectrograms](https://en.wikipedia.org/wiki/Mel_scale) |
| Perch  | 1280              | ✅ | ✅ | [Perch](https://www.nature.com/articles/s41598-023-49989-z.epdf) |
| Perch2.0  | 1536              | ✅ | ✅ | [Perch2.0](https://arxiv.org/pdf/2508.04665) |
| SurfPerch  | 1280              | ✅ | ✅ | [SurfPerch](https://arxiv.org/pdf/2404.16436) |
| Wav2Vec  | 768              | ✅ | ✅ | [Wav2Vec](https://arxiv.org/pdf/2006.11477) |
| WavLM     | 768              | ✅ | ✅ | [WavLM](https://arxiv.org/abs/2110.13900) 

### ⚠️ To add a new model:

Implement a prediction_<MODEL>() function.

Register it inside the extract_features() function in main.py.

## 📂 Datasets

This project has been evaluated on the following publicly available underwater acoustic datasets:

### DeepShip
- **Description:** Large-scale underwater acoustic dataset for ship classification, covering multiple vessel types and recording conditions.
- **Access:** https://github.com/irfankamboh/DeepShip
- **Paper:**  
  K. Irfan et al., *DeepShip: A Large-Scale Underwater Acoustic Benchmark Dataset*, IEEE OCEANS 2018.

### ShipsEar
- **Description:** Real-world underwater acoustic recordings of ships and ambient noise, collected in the port of Vigo, Spain.
- **Access:** https://atlanttic.uvigo.es/underwaternoise/ships-ear/
- **Paper:**  
  M. Santos-Domínguez et al., *ShipsEar: An Underwater Vessel Noise Database*, Applied Acoustics, 2016.

> ⚠️ **Note:**  
> Please refer to the original dataset websites for licensing terms and usage restrictions.  
> Some datasets may require registration or approval for access.

## 🔍 Notes & Best Practices

- Always use absolute paths for datasets to avoid silent errors

- The script performs directory sanity checks at runtime

