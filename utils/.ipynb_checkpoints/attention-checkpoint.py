from torch.nn import Linear, Module, BatchNorm2d, BatchNorm1d, Conv2d, Sequential, Conv1d, Tanh, LayerNorm
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from fast_transformers.events import EventDispatcher, QKVEvent


class AttentionConv2DLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, kernel_size=5, in_channels=1, event_dispatcher=""):
        super(AttentionConv2DLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or d_model #(d_model//n_heads)
        d_values = d_values or d_model #(d_model//n_heads)
        padding = (kernel_size-1)//2
        self.inner_attention = attention

        self.query_projection = Sequential(
            Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            BatchNorm2d(16),
            
            Rearrange('b c w h -> b w h c'),
            Linear(16, n_heads, bias=False),
            Rearrange('b w h a -> b w a h', a=n_heads),
            )
        self.key_projection = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=(2,1), padding=padding, bias=False),
            BatchNorm2d(16),

            Rearrange('b c w h -> b w h c'),
            Linear(16, n_heads, bias=False),
            Rearrange('b w h a -> b w a h', a=n_heads),
            )
        self.value_projection = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=(2,1), padding=padding, bias=False),
            BatchNorm2d(16),
            
            Rearrange('b c w h -> b w h c'),
            Linear(16, n_heads, bias=False),
            Rearrange('b w h a -> b w a h', a=n_heads),
            )
            
        self.out_projection = Linear(d_values * n_heads, d_model, bias=False)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, D = queries.shape
        _, S, D = keys.shape
        H = self.n_heads
        # Project the queries/keys/values
        
        queries, keys, values = queries.unsqueeze(1), keys.unsqueeze(1), values.unsqueeze(1)

        queries = self.query_projection(queries).view(N, -1, H, D)
        keys = self.key_projection(keys).view(N, -1, H, D)
        values = self.value_projection(values).view(N, -1, H, D)

        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, H*D)
        # Project the output and return
        return self.out_projection(new_values)



class AttentionConv1DLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, kernel_size=7, dilation_exp=0, event_dispatcher="",):
        super(AttentionConv1DLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or  (d_model//n_heads)
        d_values = d_values or  (d_model//n_heads)
        self.inner_attention = attention
        
        dilation = 2**dilation_exp
        padding = dilation * (kernel_size-1)//2

        self.query_projection = Sequential(
            LayerNorm(d_model),
            Rearrange('N L D -> N D L'),
            Conv1d(in_channels=d_model, out_channels= d_model*n_heads, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=d_model, bias=True),
            # BatchNorm1d(d_model),
            Rearrange('N D L -> N L D'),
            Linear(d_model*n_heads, d_keys * n_heads),
            )
        self.key_projection = Sequential(
            LayerNorm(d_model),
            Rearrange('N L D -> N D L'),
            Conv1d(in_channels=d_model, out_channels= d_model*n_heads, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=d_model, bias=True),
            # BatchNorm1d(d_model),
            Rearrange('N D L -> N L D'),
            
            Linear(d_model*n_heads, d_keys * n_heads),
            )
        
        self.mask_projection = Sequential(
            Rearrange('N L D -> N D L'),
            BatchNorm1d(d_model),
            Conv1d(in_channels=d_model, out_channels= d_model*n_heads, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=d_model, bias=True),
            
            Rearrange('N D L -> N L D'),
            Linear(d_model*n_heads, d_keys * n_heads),
            )
        
        """self.value_projection = Sequential(
            Rearrange('N L D -> N D L'),
            BatchNorm1d(d_model),
            Conv1d(in_channels=d_model, out_channels= d_model*n_heads, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1, groups=d_model, bias=True),
            
            Rearrange('N D L -> N L D'),
            Linear(d_model*n_heads, d_keys * n_heads),
            )"""
        self.value_projection = Linear(d_model, d_keys * n_heads)
        self.mask_out = Linear(d_keys * n_heads,d_keys * n_heads,)
        self.out_projection = Linear(d_keys * n_heads, d_model, bias=True)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
    def forward(self, queries, keys, values, masks, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, D = queries.shape
        _, S, D = keys.shape
        H = self.n_heads
        # Project the queries/keys/values


        queries = self.query_projection(queries).view(N, -1, H, D)
        keys = self.key_projection(keys).view(N, -1, H, D)
        values = self.value_projection(values).view(N, -1, H, D)
        masks = self.mask_projection(masks).view(N, -1, H, D)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            masks,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)
        # Project the output and return
        new_values = self.mask_out(new_values)
        return self.out_projection(F.tanh(new_values) * values.view(N, L, -1))