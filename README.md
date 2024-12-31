# Se-Inter

## Required Environment

1.CentOS release 7.2

2.[Phython 3.8 or later](https://www.python.org/)

3.[Biopython](https://biopython.org/)

4.[esm](https://github.com/facebookresearch/esm)

5.[numpy](https://numpy.org/)

6.[hh-suite](https://github.com/soedinglab/hh-suite)

## Installation

#### 1.Install DRN-1D2D_Inter

```
git clone https://github.com/HQY6324/se-inter.git
```
## Data Preparation

#### 1.Create the FASTA files for PPI and the corresponding paired FASTA files separately.The corresponding scripts `makesinglefastas.py` and `makepairfastas.py` for generating FASTA files are provided in the `bin` directory.

#### 2.Generate the single MSA and paired MSA separately. The corresponding scripts `singlemsaprod.py` and `pairmsaprod.py` for creating the MSAs are provided in the `bin` directory. Users need to prepare the corresponding UniRef30_2021_03 database files and adjust the input and output folder paths accordingly.

#### 3.ESM package and ESM-MSA pre-trained model for producing ESM-MSA features.You should install your owm [ESM](https://github.com/facebookresearch/esm) or (pip install fair-esm).You may also need to download the pretrained model [esm_msa1_t12_100M_UR50S.pt](https://github.com/facebookresearch/esm) and the regression model [esm_msa1_t12_100M_UR50S-contact-regression.pt](https://dl.fbaipublicfiles.com/fair-esm/regression/esm_msa1_t12_100M_UR50S-contact-regression.pt). The scripts for generating ESM-MSA features are located in the `bin` directory, and users need to adjust the corresponding paths themselves.

#### 4.Users can download `SaProt_650M_AF2.pt` from [SaProt_650M_AF2.pt](https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/SaProt_650M_AF2.pt)

#### 5.Use the Foldseek tool to generate structure-aware sequences. Some residues may have structural gaps, and manual alignment with the FASTA sequence is required. After sequence alignment, use Saprot-650M to generate features.The related script files are located in the `bin` directory.

#### 6.The script for generating the labeled protein contact matrix is ready. Users need to prepare the two monomer PDB files before generating the label matrix.

## Usage

#### After preparing all the feature and label files, modify the corresponding feature paths and output file save paths in the `predict.py` file, then execute the script. An example is provided in the `example` folder, and you can use the features in this folder for prediction.

### Example

```
python predict.py 
```

### Train

#### After preparing all the features required for training, use the `featmakepkl.py` script to serialize them into a single `.pkl` file. This `.pkl` file will be used for training.

#### The script used to train Se-Inter is `train.py`, which contains all the details of training Se-Inter, including how to choose the best model, how to calculate the loss, etc.
