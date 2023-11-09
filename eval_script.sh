#!/usr/bin/env bash

python eval_script.py --src cs --tgt en --batch-size 16
echo "cs-en done"
python eval_script.py --src de --tgt en --batch-size 16
echo "de-en done"
python eval_script.py --src fi --tgt en --batch-size 16
echo "fi-en done"
python eval_script.py --src ro --tgt en --batch-size 16
echo "ro-en done"
python eval_script.py --src ru --tgt en --batch-size 16
echo "ru-en done"
python eval_script.py --src tr --tgt en --batch-size 16
echo "tr-en done"


