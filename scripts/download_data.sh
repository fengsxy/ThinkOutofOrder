#!/bin/bash
# Download datasets for order robustness experiments

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$DATA_DIR"/{gsm8k,math500,reasonorderqa}

echo "Downloading GSM8K..."
if [ ! -f "$DATA_DIR/gsm8k/test.jsonl" ]; then
    python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('openai/gsm8k', 'main')
for split in ['train', 'test']:
    with open('$DATA_DIR/gsm8k/{}.jsonl'.format(split), 'w') as f:
        for ex in ds[split]:
            f.write(json.dumps({'question': ex['question'], 'answer': ex['answer']}) + '\n')
print('GSM8K downloaded successfully')
"
else
    echo "GSM8K already exists, skipping..."
fi

echo "Downloading MATH500..."
if [ ! -f "$DATA_DIR/math500/test.jsonl" ]; then
    python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
with open('$DATA_DIR/math500/test.jsonl', 'w') as f:
    for ex in ds:
        f.write(json.dumps({
            'problem': ex['problem'],
            'solution': ex['solution'],
            'answer': ex.get('answer', ''),
            'level': ex.get('level', ''),
            'type': ex.get('type', '')
        }) + '\n')
print('MATH500 downloaded successfully')
"
else
    echo "MATH500 already exists, skipping..."
fi

echo "ReasonOrderQA should be generated or provided separately."
echo "See data/reasonorderqa/README.md for instructions."

echo ""
echo "Data download complete!"
echo "  GSM8K: $DATA_DIR/gsm8k/"
echo "  MATH500: $DATA_DIR/math500/"
echo "  ReasonOrderQA: $DATA_DIR/reasonorderqa/"
