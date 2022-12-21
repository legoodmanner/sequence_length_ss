#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement transformer encoders and decoders that are going to be used with
different attention mechanisms.

In all cases the batch dimension is first and the sequence dimension is second.
"""

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, BatchNorm2d, BatchNorm1d, Sequential, Conv1d
import torch.nn.functional as F

from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask
from einops.layers.torch import Rearrange

class TransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, out_d_model=None, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        out_d_model = out_d_model or d_model
        self.attention = attention
        self.linear1 = Linear(out_d_model, out_d_model*4)
        self.linear2 = Linear(out_d_model*4, out_d_model)
        """ self.linear1 = Sequential(
            Rearrange('b w h -> b h w'),
            Conv1d(out_d_model, out_d_model, 3, 1, 1, 1, out_d_model),
            Conv1d(out_d_model, out_d_model//4, 1, 1, 0, 1),
            Rearrange('b h w -> b w h'),
        ) 
        self.linear2 = Linear(out_d_model//4, out_d_model)"""
        # self.norm1 =  BatchNorm2d(1)# LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)  #BatchNorm2d(1) #LayerNorm(d_model)
        self.norm1 =  LayerNorm(d_model)  #BatchNorm2d(1)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.silu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, softmax_temp, pos_k=None, source_indice=None, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        

        # Run self attention and add it to the input 
        # x = self.norm1(x.unsqueeze(1)).squeeze(1)
        x =  x + self.dropout(self.attention(
            x, x, x, x,
            pos_k = pos_k,
            source_indice=source_indice,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask,
            softmax_temp = softmax_temp,
        ))
        # Run the fully connected part of the layer
        # y = x.clone()
        y = x = self.norm1(x)
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.linear2(x))

        # return self.norm2((x+y).unsqueeze(1)).squeeze(1)
        return self.norm2(x + y)
        # return x+y
        """ y = x = self.norm1(x.unsqueeze(1))
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y).squeeze() """


class TransformerEncoder(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, softmax_temp=None, source_indice=None, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, softmax_temp, source_indice=source_indice, attn_mask=attn_mask, length_mask=length_mask)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoderLayer(Module):
    """The decoder layer from "Attention Is All You Need".

    Similar to the encoder layer, this layer implements the decoder that
    PyTorch implements but can be used with any attention implementation
    because it receives the attention layers as constructor arguments.

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
                        given as a nn.Module
        cross_attention: The attention implementation to use for cross
                         attention given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, self_attention, cross_attention, d_model, out_d_model=None, d_ff=None,
                 dropout=0.1, activation="relu", event_dispatcher=""):
        super(TransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        out_d_model = out_d_model or d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = Linear(out_d_model, d_ff)
        self.linear2 = Linear(d_ff, out_d_model)
        self.norm1 =  BatchNorm1d(d_model)
        self.norm2 =  BatchNorm1d(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        """Apply the transformer decoder to the input x using the memory
        `memory`.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E should be the same as
               the d_model passed in the constructor.
            memory: The memory features of shape (N, L', E) where N is the
                    batch size, L' is the memory's sequence length (padded) and
                    E should be the same as the d_model.
            x_mask: An implementation of fast_transformers.masking.BaseMask
                    that encodes where each element of x can attend to in x.
                    Namely the self attention mask.
            x_length_mask: An implementation of a BaseMask that encodes how
                           many elements each sequence in the batch consists
                           of.
            memory_mask: An implementation of BaseMask that encodes where each
                         element of x can attend to in the memory. Namely the
                         cross attention mask.
            memory_length_mask: An implementation of a BaseMask that encodes how
                                many elements each memory sequence in the batch
                                consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = self.norm1(x.permute(0,2,1)).permute(0,2,1)
        x = x + self.dropout(self.self_attention(
            x, x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))

        x = self.norm2(x.permute(0,2,1)).permute(0,2,1)
        # Secondly apply the cross attention and add it to the previous output
        x = self.dropout(self.cross_attention(
            memory, x, memory, x,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))

        # Finally run the fully connected part of the layer
        y = x.clone()
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return x+y


class TransformerDecoder(Module):
    """TransformerDecoder is little more than a sequence of transformer decoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ----------
        layers: list, TransformerDecoderLayer instances or instances that
                implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # Apply all the transformer decoders
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask,
                      memory_length_mask=memory_length_mask)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x
