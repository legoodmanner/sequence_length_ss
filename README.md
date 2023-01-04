# How Long Can it Separate ?
This is the experimental repo regarding source separation on long sequence of audio. We examined whether the long sequence of the input could be beneficial to the performance of source separation with self-attention.

## Motiviation
Anticipating the model can make stronger reference to the partial clips with relatively prominent sounds of the targets, we used self-attention to capture those distanced but helpful reference with logner input.
## Usage
1. Before training, some hyperparameters need to be determined in `train_transformer.sh`:
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
- `src`: The stem to train on. There are 4 options:
    - `drums`
    - `bass`
    - `other`
    - `vocals`:
2. Use the command below to train the transformer separator:
```
bash train_transformer.sh
```
