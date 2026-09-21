import argparse
import json
import traceback
from pathlib import Path

from .core import ModelBackend, environment, save_json, seed_all
from .controls import reinjection_control, rope_control
from .data import prepare_tokens, extract_pairs


class ExperimentRunner:
    def __init__(self,cfg):
        self.cfg=cfg
        self.out=Path(cfg['output'])
        from .identity import protect_pilot, config_identity, bind
        protect_pilot(self.out)
        previous=self.out/'config.json'
        if previous.exists() and config_identity(json.loads(previous.read_text())) != config_identity(cfg):
            raise ValueError('Configuration differs from the saved run; choose a new output directory')
        bind(self.out/'run_manifest.json', dict(schema=2, config=config_identity(cfg)))
        seed_all(cfg['seed'])
        if not previous.exists(): save_json(previous,cfg)

    def backend(self,role):
        return ModelBackend(self.cfg[role],self.cfg['device'],self.cfg['dtype'],self.cfg['attention'],
                            self.cfg.get(role+'_revision'))

    def run(self,stage):
        cfg=self.cfg
        if stage=='environment':
            save_json(self.out/'environment.json',environment())
            print(json.dumps(environment(),indent=2))
            return
        if stage in ['introspect','control','extract']:
            target=self.backend('target')
            if stage=='control':
                reinjection_control(cfg,target)
                return
            source=self.backend('source')
            if stage=='introspect':
                s,t=source.introspect(),target.introspect()
                result={'source':s,'target':t,
                    'vocab_identical':source.tokenizer.get_vocab()==target.tokenizer.get_vocab(),
                    'tokenizer_backend_identical':s['backend_tokenizer_sha256']==t['backend_tokenizer_sha256'],
                    'vocab_dimensions_identical':s['vocab_size']==t['vocab_size'],
                    'special_tokens_identical':s['special_tokens']==t['special_tokens']}
                save_json(self.out/'introspection.json',result)
                assert result['vocab_identical'] and result['tokenizer_backend_identical'] and result['vocab_dimensions_identical']
                rope_control(cfg,source,'source')
                rope_control(cfg,target,'target')
                print(json.dumps(result,indent=2))
                return
            assert (self.out/'controls_passed.json').exists(), 'Run passing same-model controls first'
            root=prepare_tokens(cfg,target.tokenizer)
            extract_pairs(cfg,source,target,root)
            return
        if stage in ['train','evaluate','reasoning','stretch','refine','latency','reconstruction']:
            assert (self.out/'controls_passed.json').exists()
            source,target=self.backend('source'),self.backend('target')
            root=prepare_tokens(cfg,target.tokenizer)
            if stage=='train':
                from .mapping import train_mappers
                train_mappers(cfg,source,target,root)
            elif stage=='evaluate':
                from .evaluation import evaluate
                evaluate(cfg,source,target,root)
            elif stage=='reasoning':
                from .reasoning import benchmark
                benchmark(cfg,source,target)
            elif stage=='stretch':
                from .functional_training import train_functional
                train_functional(cfg,source,target,root)
            elif stage=='refine':
                from .data import validate_training_caches
                validate_training_caches(cfg,source,target,root)
                from .refinement import refine
                refine(cfg,source,target,root)
            elif stage=='latency':
                import numpy as np
                from .mapping import CacheAdapter
                from .evaluation import latency_sweep
                latency_sweep(cfg,source,target,CacheAdapter.load(cfg,source,target),np.load(root/'test_tokens.npy'))
            elif stage=='reconstruction':
                from .evaluation import evaluate_reconstruction
                evaluate_reconstruction(cfg,source,target,root)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['environment','introspect','control','extract','train','evaluate','reasoning','stretch','refine','latency','reconstruction'])
    p.add_argument('--config',default='configs/qwen3_1.7b_to_0.6b.json')
    p.add_argument('--source')
    p.add_argument('--target')
    p.add_argument('--device')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    for name in ['source','target','device']:
        if getattr(args,name): cfg[name]=getattr(args,name)
    runner=ExperimentRunner(cfg)
    try:
        runner.run(args.stage)
    except Exception:
        failure=runner.out/'failures.jsonl'
        with failure.open('a') as f:
            f.write(json.dumps(dict(stage=args.stage,traceback=traceback.format_exc()))+'\n')
        raise
