from datasets import  load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BatchEncoding
from transformers import Trainer, TrainingArguments
import pandas as pd
import time
import gc
import argparse
import torch 
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Arguments
def main():
  parser = argparse.ArgumentParser(description="Arguments for the function")
  parser.add_argument("--src", type=str, help="src language include one of cs, de, en, fi, ro, ru, tr")
  parser.add_argument("--tgt", type=str, help="tgt language include only en for now")
  parser.add_argument("--batch-size", type=int, help="tgt language include only en for now")
  args = parser.parse_args()

  #Loading the input data
  codes = {'cs':'cs_CZ', 'de':'de_DE', 'en': 'en_XX', 'fi': 'fi_FI',  'ro': 'ro_RO', 'ru':'ru_RU', 'tr':'tr_TR'}
  src_lang = args.src
  targ_lang = args.tgt
  ds = load_dataset('wmt16',f'{src_lang}-{targ_lang}')


  model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
  model = model.to(device)
  tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")



  source_text = []
  target_text = []
  translated_text = []
  batch_size = args.batch_size
  import time
  import gc
  for i in range(0,len(ds['validation']),batch_size):
    start = time.time()

    # input data
    df = pd.DataFrame(ds['validation']['translation'][i:i+batch_size])
    current_batch_source = list(df[src_lang])
    target_batch_target = list(df[targ_lang])
    source_text += current_batch_source
    target_text += target_batch_target
    tokenizer.src_lang = codes[src_lang]

    # model decode
    encoded_src = tokenizer(current_batch_source, return_tensors="pt", padding='longest')
    encoded_src = encoded_src.to(device)
    
    generated_tokens = model.generate(**encoded_src, forced_bos_token_id=tokenizer.lang_code_to_id[codes[targ_lang]],pad_token_id=1)
    batch_result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_text += batch_result
    print('time taken', time.time()-start)
    # clearing cache
    torch.cuda.empty_cache()
    encoded_src = None
    gc.collect()
    
  
  with open(f'reference_{src_lang}_{targ_lang}.txt','w') as fp:
    fp.write('\n'.join(target_text))


  with open(f'translated_{src_lang}_{targ_lang}.txt','w') as fp:
    fp.write('\n'.join(translated_text))

if __name__ == "__main__":
    main()
