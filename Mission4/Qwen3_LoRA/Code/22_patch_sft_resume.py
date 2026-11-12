"""Patch the existing, server-side LIST SFT script without changing its prompts or examples.
Run with the same Python that has all dependencies, or system python3 (stdlib only).
"""
from pathlib import Path
import ast

ROOT = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
src = ROOT / 'Code/20_qwen_sft_pilot_4096.py'
dst = ROOT / 'Code/22_qwen_sft_resume.py'
assert src.is_file(), f'Missing original source: {src}'
assert not dst.exists(), f'Refusing to overwrite: {dst}'
s = src.read_text(encoding='utf-8')

def once(old, new):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'Expected once, found {n}: {old[:110]!r}')
    s = s.replace(old,new,1)

once('import argparse\n', 'import argparse\nimport os\nimport shutil\n')
once('from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training',
     'from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training')
once("    p.add_argument('--smoke',action='store_true'", "    p.add_argument('--save-every',type=int,default=256,help='optimizer steps per checkpoint')\n    p.add_argument('--resume',action='store_true',help='resume latest complete checkpoint')\n    p.add_argument('--smoke',action='store_true'")
once('    random.seed(args.seed)\n', '''    assert args.save_every > 0
    if args.resume and args.smoke: raise ValueError('--resume and --smoke cannot be combined')
    random.seed(args.seed)
''')
once('    model=get_peft_model(model,lora)\n', '''    ckpt_root=OUT / f'run_{args.train_limit}_resume'
    ckpt_root.mkdir(parents=True,exist_ok=True)
    # Strictly use a complete checkpoint (state.pt written before final rename).
    snapshots=sorted((p for p in ckpt_root.glob('step_*')
                      if p.is_dir() and (p/'state.pt').is_file()
                      and (p/'adapter'/'adapter_config.json').is_file()),
                     key=lambda p:int(p.name.split('_')[-1]))
    latest=snapshots[-1] if snapshots else None
    if args.resume and latest is None:
        raise RuntimeError(f'No resumable checkpoint found under {ckpt_root}')
    if not args.resume and latest is not None:
        raise RuntimeError(f'Existing checkpoints found at {ckpt_root}; use --resume')
    if args.resume:
        model=PeftModel.from_pretrained(model,str(latest/'adapter'),is_trainable=True)
        print(f'[RESUME] loading {latest}',flush=True)
    else:
        model=get_peft_model(model,lora)
''')
once('    order=list(range(len(records)))\n    random.shuffle(order)\n    model.train()\n', '''    order=list(range(len(records)))
    random.shuffle(order)
    start_i=0
    if args.resume:
        state=torch.load(latest/'state.pt',map_location='cpu',weights_only=False)
        expected={'train_limit':args.train_limit,'seed':args.seed,
                  'max_length':args.max_length,'lr':args.lr,
                  'grad_accum':args.grad_accum}
        if state['config'] != expected:
            raise RuntimeError(f'Configuration mismatch: saved={state["config"]}, requested={expected}')
        opt.load_state_dict(state['optimizer'])
        order=state['order']
        start_i=int(state['examples_done'])
        assert len(order)==len(records) and start_i%args.grad_accum==0
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'[RESUME] restored examples_done={start_i} optimizer_step={start_i//args.grad_accum}',flush=True)
    model.train()
''')
once('    for i,j in enumerate(order,1):\n', '    for i in range(start_i+1,len(order)+1):\n        j=order[i-1]\n')
once('            opt.step();opt.zero_grad(set_to_none=True)\n', '''            opt.step();opt.zero_grad(set_to_none=True)
            update_idx=(i+args.grad_accum-1)//args.grad_accum
            if i%args.grad_accum==0 and update_idx%args.save_every==0:
                final=ckpt_root/f'step_{update_idx:07d}'
                stage=ckpt_root/f'.step_{update_idx:07d}.tmp'
                if final.exists():raise FileExistsError(final)
                if stage.exists():shutil.rmtree(stage)
                stage.mkdir(parents=True)
                model.save_pretrained(str(stage/'adapter'))
                config={'train_limit':args.train_limit,'seed':args.seed,
                        'max_length':args.max_length,'lr':args.lr,
                        'grad_accum':args.grad_accum}
                torch.save({'config':config,'examples_done':i,'order':order,
                            'optimizer':opt.state_dict(),
                            'python_rng':random.getstate(),
                            'cpu_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state_all()},stage/'state.pt')
                os.replace(stage,final)
                print(f'[CHECKPOINT] {final} examples_done={i}',flush=True)
                # Retain last two complete checkpoints to bound disk use.
                complete=sorted((p for p in ckpt_root.glob('step_*')
                                if p.is_dir() and (p/'state.pt').is_file()),
                               key=lambda p:int(p.name.split('_')[-1]))
                for old in complete[:-2]:shutil.rmtree(old)
''')
# avoid reusing final adapter if it already exists, but ensure check happens before costly training
once('    model.train()\n    losses=[]', '''    adapter=OUT / ('smoke_adapter' if args.smoke else f'pilot_{args.train_limit}_adapter')
    if adapter.exists():raise FileExistsError(f'Final adapter already exists: {adapter}')
    model.train()
    losses=[]''')
ast.parse(s)
dst.write_text(s,encoding='utf-8')
print('[CREATED]',dst)
print('[PRESERVED] Original prompts, targets, training data sampling and evaluation')
print('[CHECKPOINT] Every 256 optimizer steps by default; keep two completed snapshots')
