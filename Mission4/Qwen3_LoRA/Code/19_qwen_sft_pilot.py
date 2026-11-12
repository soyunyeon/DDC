"""Qwen3-4B instruction-style causal LM QLoRA pilot, single GPU, no future utterances.
Usage: CUDA_VISIBLE_DEVICES=0 python -u 19_qwen_sft_pilot.py --train-limit 512 --val-limit 80
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
MODEL = ROOT / 'Models/Qwen3-4B'
TRAIN = ROOT / 'Data/prefix_train_8.jsonl'
VAL = ROOT / 'Data/prefix_val_all.jsonl'
OUT = ROOT / 'Checkpoints/qwen3_4b_sft_pilot'
PREDS = ROOT / 'Predictions/qwen3_4b_sft_pilot'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
SYSTEM = ("너는 119 신고 대화의 환자 증상을 분류하는 분석기다. 현재까지 나온 대화만 참고한다. "
          "가능한 증상은 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란이다. "
          "반드시 이 증상들의 한국어 이름을 사용한 JSON 배열 한 개만 출력하라. "
          "설명, 추론 과정, Markdown은 출력하지 마라. 정보가 부족하면 가능한 증상 중 근거가 있는 것만 예측하라.")


def read_rows(path, n):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if n and len(rows)>=n:break
    return rows


def prompt_tokens(tok, row, max_input):
    messages=[{'role':'system','content':SYSTEM},
              {'role':'user','content':'현재까지의 119 신고 대화:\n'+row['text']}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    ids=tok(prompt,add_special_tokens=False)['input_ids']
    if len(ids)>max_input:
        # Preserve the assistant-generation header; truncate dialogue at utterance boundaries.
        lines=row['text'].split('\n')
        while len(ids)>max_input and len(lines)>1:
            lines.pop()
            messages[1]['content']='현재까지의 119 신고 대화:\n'+'\n'.join(lines)
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok(prompt,add_special_tokens=False)['input_ids']
        if len(ids)>max_input:
            # Very long first utterance: preserve header while shorten user content.
            first=lines[0]
            while len(ids)>max_input and len(first)>16:
                first=first[:max(16,int(len(first)*0.8))]
                messages[1]['content']='현재까지의 119 신고 대화:\n'+first
                prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok(prompt,add_special_tokens=False)['input_ids']
            if len(ids)>max_input:raise ValueError('Prompt instruction exceeds context length')
    return ids


def target_tokens(tok, row):
    names=[name for name,flag in zip(LABELS,row['label_vector']) if flag]
    return tok(json.dumps(names,ensure_ascii=False,separators=(',',':')),
               add_special_tokens=False)['input_ids']+[tok.eos_token_id]


def make_example(tok,row,max_len):
    target=target_tokens(tok,row)
    if len(target)>=max_len:raise ValueError('Output exceeds context length')
    inp=prompt_tokens(tok,row,max_len-len(target))
    ids=inp+target
    return {'input_ids':ids,'labels':[-100]*len(inp)+target,
            'attention_mask':[1]*len(ids)}


def parse_prediction(s):
    s=s.strip()
    try:
        names=json.loads(s)
        if isinstance(names,list) and all(isinstance(x,str) and x in LABELS for x in names):
            return list(dict.fromkeys(names)),True
    except (ValueError,TypeError):pass
    return [],False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-limit',type=int,default=512)
    p.add_argument('--val-limit',type=int,default=80)
    p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true',help='test with 16 training rows and 8 val rows')
    args=p.parse_args()
    if args.smoke:
        args.train_limit=16;args.val_limit=8
    assert 0<args.train_limit<=20000 and 0<args.val_limit<=1000
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(),'CUDA GPU required'
    torch.cuda.set_device(0)
    OUT.mkdir(parents=True,exist_ok=True);PREDS.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    tok.pad_token=tok.eos_token
    tok.padding_side='right'
    all_train=read_rows(TRAIN,0)
    rng=random.Random(args.seed)
    sample=rng.sample(all_train,args.train_limit)
    val=read_rows(VAL,0)
    # Deterministic, call-level validation subset, never used for training.
    rng=random.Random(args.seed)
    rng.shuffle(val)
    val=val[:args.val_limit]
    print(f'[DATA] train={len(sample)} val={len(val)}',flush=True)
    records=[make_example(tok,r,args.max_length) for r in sample]
    print('[TOKEN] max_sequence_len=',max(len(x['input_ids']) for x in records),flush=True)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    print('[LOAD] Qwen3-4B causal LM, non-thinking instruction format',flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,
                quantization_config=cfg,device_map={'':0},dtype=torch.bfloat16,
                attn_implementation='sdpa')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                    target_modules=['q_proj','k_proj','v_proj','o_proj',
                                    'gate_proj','up_proj','down_proj'],
                    task_type=TaskType.CAUSAL_LM,bias='none')
    model=get_peft_model(model,lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=args.lr)
    order=list(range(len(records)))
    random.shuffle(order)
    model.train()
    losses=[];start=time.time(); opt.zero_grad(set_to_none=True)
    for i,j in enumerate(order,1):
        batch={key:torch.tensor([records[j][key]],device='cuda:0') for key in ('input_ids','labels','attention_mask')}
        loss=model(**batch).loss
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at example {i}')
        (loss/args.grad_accum).backward()
        losses.append(float(loss.item()))
        if i%args.grad_accum==0 or i==len(order):
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(grad):raise RuntimeError(f'nonfinite gradient at example {i}')
            opt.step();opt.zero_grad(set_to_none=True)
        if i==1 or i%64==0 or i==len(order):
            print(f'[TRAIN] {i}/{len(order)} loss={loss.item():.4f} avg_recent={sum(losses[-64:])/len(losses[-64:]):.4f} '
                  f'peak_mem={torch.cuda.max_memory_reserved()/2**30:.2f}GiB elapsed={time.time()-start:.0f}s',flush=True)
    adapter=OUT / ('smoke_adapter' if args.smoke else f'pilot_{args.train_limit}_adapter')
    if adapter.exists():raise FileExistsError(f'Will not overwrite existing adapter: {adapter}')
    model.save_pretrained(str(adapter));tok.save_pretrained(str(adapter))
    print('[SAVED]',adapter,flush=True)
    model.eval();model.config.use_cache=True
    rows=[];valid=0;score_sum=0
    path=PREDS/('smoke_predictions.jsonl' if args.smoke else f'pilot_{args.train_limit}_predictions.jsonl')
    with torch.inference_mode(),path.open('w',encoding='utf-8') as f:
        for k,r in enumerate(val,1):
            ids=prompt_tokens(tok,r,args.max_length)
            input_ids=torch.tensor([ids],device='cuda:0')
            generated=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=args.max_new_tokens,do_sample=False,
                        pad_token_id=tok.eos_token_id)
            raw=tok.decode(generated[0,len(ids):],skip_special_tokens=True).strip()
            pred,ok=parse_prediction(raw)
            gold={name for name,flag in zip(LABELS,r['label_vector']) if flag}
            pred_set=set(pred)
            score=len(gold&pred_set)/len(gold|pred_set) if gold|pred_set else 1.0
            score_sum+=score;valid+=int(ok)
            f.write(json.dumps({'call_id':r['call_id'],'step':r['step'],
                     'predicted':pred,'gold':list(gold),'valid_json':ok,
                     'raw_output':raw[:300],'jaccard':score},ensure_ascii=False)+'\n')
            if k==1 or k%20==0 or k==len(val):
                print(f'[EVAL] {k}/{len(val)} valid_json={valid}/{k} mean_prefix_jaccard={score_sum/k:.4f}',flush=True)
    print(f'[DONE] train={len(sample)} val={len(val)} valid_json_rate={valid/len(val):.3f} '
          f'prefix_jaccard={score_sum/len(val):.4f} output={path}',flush=True)
    print('[NOTE] This small random prefix diagnostic is NOT official per-call validation score.',flush=True)

if __name__=='__main__':main()
