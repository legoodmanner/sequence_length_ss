# How Long Can it Separate ?
This is the experimental repo regarding source separation on long sequence of audio. We examined whether the long sequence of the input could be beneficial to the performance of source separation with self-attention.

## Motiviation
Anticipating the model can make stronger reference to the partial clips with relatively prominent sounds of the targets, we used self-attention to capture those distanced but helpful reference with logner input.
## Usage
1. **Before training, some hyperparameters need to be determined in `train_transformer.sh`:**
- `cuda`: The id of GPU to train the models
- `separator`: The types of the separators. There are three options:
    - `umx`: The original design of Open-Unmix. [1]
    - `umx_transformer`: The rivesd Transformer version of the Open-Unmix, supplanting the RNN with the Transformers.
    - `auto_transformer`: The revised Transformer version of the U-Net [2], supplanting the CNN-encoders/decoers with the Transformers.
- `attention`: The types of the self-attention layers. Again, there are three options:
    - `qkv2D`: The self-attention using Conv2D to generate the 'queries', 'keys', and 'values'
    - `qkv1D`: The self-attention using Conv1D to generate the 'queries', 'keys', and 'values'
    - `qkvg1D`: The self-attention using Conv2D to generate the 'queries', 'keys', 'values', and additional 'gates'.
- `bs`: The batch size
- `src`: The stem to train on. There are 4 options: `drums`, `bass`, `other`, and `vocals`
- `ckpt_dir`: The path of the directory for model checkpoint
2. **Use the command below to train the transformer separator:**
```
bash train_transformer.sh
```
3. **For testing, again, please determine the hyperparameters in script of `test.sh` beforehand**
```
bash test.sh 
```

## Dataset
We used MUSD18 [3] to train and test our source separation model and followed the pre-defined training set, validation set, and testing set as our subsets during our experiements.

## Evaluation
The signal distortion ration (SDR) and signal-to-noise ratio (SNR) are calculated based on the `mireval` proposed in [4]

## Reference
- [1] F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji, Open-Unmix - A Reference Implementation for Music Source Separation, 2019
- [2] A. Jansson and E. J. Humphrey and N Montecchio and R Bittner and A Kumar and T Weyde, Singing voice separation with deep U-Net convolutional networks, 2017
- [3] [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)
- [4] [museval](https://github.com/sigsep/sigsep-mus-eval)
