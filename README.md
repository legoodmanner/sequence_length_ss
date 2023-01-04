# How Long Can it Separate ?
This is the experimental repo regarding source separation on long sequence of audio. We examined whether the long sequence of the input could be beneficial to the performance of source separation with self-attention.

## Abstract
An audio recording of music usually contains multiple sections that are each played by a different subset of instruments, or sound sources, that compose the piece. Each occurrence of a sound source may co-occur with different sound sources each time, but all the occurrences of the same sound source typically share the same timbre. To segregate a certain sound source from the sound mixture of the audio recording into a separate audio stream, it might be useful for the network to have a receptive field that is long enough to have access to multiple occurrences of the sound source for cross references. We are curious whether longer segmented input sequence of mixing audio is able to benefit the separation result as increase of reception field to the single song or being counterproductive under different setting. We present an empirical investigation of the effect for varying length of input sequences testing on various neural network structures. By resulting metrics such as signal-to-distortion ratio (SDR) of blind source separation, the tendency of result between different length of input, different instrument sources and different architecture of models would be explicitly shown and compared. We use MUSDB dataset for model training and evaluation and experiment with varying the length of the input spectrogram from 7 seconds to up to 60 seconds. Lastly, we would investigate self-attention mechanics and introduce the revised self-attention with adaption to work more properly with long sequence input than the original one.  

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
- [1] F.-R. Stöter and S. Uhlich and A. Liutkus and Y. Mitsufuji, Open-Unmix - A Reference Implementation for Music Source Separation, 2019
- [2] A. Jansson and E. J. Humphrey and N Montecchio and R Bittner and A Kumar and T Weyde, Singing voice separation with deep U-Net convolutional networks, 2017
- [3] [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)
- [4] [museval](https://github.com/sigsep/sigsep-mus-eval)
