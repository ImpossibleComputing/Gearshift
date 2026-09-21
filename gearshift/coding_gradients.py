"""Checkpointed receiver continuation with immutable historical K/V inputs."""
import torch
from torch.utils.checkpoint import checkpoint

class ReadOnlyPrefix:
    """Recomputation never mutates or appends to the shared historical cache."""
    def __init__(self,k,v,index):self.k,self.v,self.index=k,v,index
    def update(self,k,v,index,cache_kwargs=None):
        if index!=self.index:raise ValueError('Receiver layer identity mismatch')
        return torch.cat((self.k,k),dim=-2),torch.cat((self.v,v),dim=-2)

def continuation(model,pairs,ids,positions=None,checkpoint_layers=True):
    if model.config.model_type!='qwen3' or model.config.rope_scaling:
        raise ValueError('Only the pinned unscaled Qwen3 architecture is verified')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Receiver language model must be frozen and in evaluation mode')
    n=pairs[0][0].shape[-2];device=pairs[0][0].device
    ids=torch.as_tensor(ids,dtype=torch.long,device=device).reshape(1,-1)
    if n+ids.shape[1]>model.config.max_position_embeddings:raise ValueError('Complete history exceeds model context')
    if len(pairs)!=len(model.model.layers):raise ValueError('Cache layer count mismatch')
    h=model.model.embed_tokens(ids)
    absolute=torch.arange(n,n+ids.shape[1],device=device)
    keys=torch.arange(n+ids.shape[1],device=device)
    allowed=keys.unsqueeze(0)<=absolute.unsqueeze(1)
    mask=torch.zeros((ids.shape[1],n+ids.shape[1]),dtype=h.dtype,device=device)
    mask.masked_fill_(~allowed,torch.finfo(h.dtype).min);mask=mask[None,None]
    rope=model.model.rotary_emb(h,absolute[None])
    for index,(block,(k,v)) in enumerate(zip(model.model.layers,pairs)):
        if k.shape[-2]!=n or v.shape!=k.shape:raise ValueError('Inconsistent historical lengths or geometry')
        def layer(x,pk,pv,block=block,index=index):
            return block.forward(x,attention_mask=mask,position_ids=absolute[None],
                past_key_values=ReadOnlyPrefix(pk,pv,index),use_cache=False,
                cache_position=absolute,position_embeddings=rope)
        h=checkpoint(layer,h,k,v,use_reentrant=False,preserve_rng_state=False) if checkpoint_layers else layer(h,k,v)
    h=model.model.norm(h)
    if positions is not None:h=h[:,positions]
    return model.lm_head(h)

class AffineMapper(torch.nn.Module):
    def __init__(self,source,target,seed=20260915):
        super().__init__();self.source,self.target=source,target
        self.sources=[round(i*(source.config.num_hidden_layers-1)/(target.config.num_hidden_layers-1)) for i in range(target.config.num_hidden_layers)]
        d=source.config.num_key_value_heads*source.config.head_dim
        out=target.config.num_key_value_heads*target.config.head_dim
        if d!=1024 or out!=1024:raise ValueError('Pilot mapper geometry differs from the approved pair')
        g=torch.Generator(device=target.device).manual_seed(seed)
        self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d,out,generator=g,device=target.device)*.001) for _ in range(2*len(self.sources))])
        self.biases=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(out,device=target.device)) for _ in self.weights])

    def forward(self,pairs):
        from gearshift.core import CacheExtractor
        result=[]
        for layer,index in enumerate(self.sources):
            mapped=[]
            for kind in range(2):
                x=pairs[index][kind]
                if kind==0:x=self.source.rope(x,inverse=True)
                x=CacheExtractor.flatten(x).float()
                y=x@self.weights[2*layer+kind]+self.biases[2*layer+kind]
                y=CacheExtractor.unflatten(y,self.target.config.num_key_value_heads,self.target.config.head_dim)
                if kind==0:y=self.target.rope(y)
                mapped.append(y.to(self.target.dtype))
            result.append(tuple(mapped))
        return tuple(result)
