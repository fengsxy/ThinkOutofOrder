#!/bin/bash
# Smoke test for order-robustness-diffusion
# Run with: bash scripts/smoke_test.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

echo "=== Order Robustness Smoke Test ==="
echo ""

# Check Python syntax for all eval scripts
echo "1. Checking Python syntax..."
python3 -m py_compile eval/run_llada_eval_gsm8k.py
python3 -m py_compile eval/run_llada_eval_reasonorderqa.py
python3 -m py_compile eval/run_dream_eval.py
python3 -m py_compile eval/run_qwen_eval.py
python3 -m py_compile data/generate_reasonorderqa.py
echo "   All scripts pass syntax check."
echo ""

# Check data files exist
echo "2. Checking data files..."
if [ -f "data/gsm8k/test.jsonl" ]; then
    echo "   GSM8K: $(wc -l < data/gsm8k/test.jsonl) problems"
else
    echo "   GSM8K: NOT FOUND (run scripts/download_data.sh)"
fi

if [ -f "data/math500/test.jsonl" ]; then
    echo "   MATH500: $(wc -l < data/math500/test.jsonl) problems"
else
    echo "   MATH500: NOT FOUND (run scripts/download_data.sh)"
fi

if [ -f "data/reasonorderqa/reasonorderqa.jsonl" ]; then
    echo "   ReasonOrderQA: $(wc -l < data/reasonorderqa/reasonorderqa.jsonl) problems"
else
    echo "   ReasonOrderQA: NOT FOUND"
fi
echo ""

# Test ReasonOrderQA generation
echo "3. Testing ReasonOrderQA generation..."
python3 data/generate_reasonorderqa.py --count 5 --output /tmp/test_roqa.jsonl --seed 42
echo "   Generated 5 test problems to /tmp/test_roqa.jsonl"
echo ""

# Check model imports (without loading weights)
echo "4. Checking model imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from models.llada import LLaDAConfig, LLaDAModelLM
    print('   LLaDA model imports OK')
except ImportError as e:
    print(f'   LLaDA import failed: {e}')
    print('   Run: pip install -r requirements.txt')
"
echo ""

# Check if .env exists
echo "5. Checking configuration..."
if [ -f ".env" ]; then
    echo "   .env file exists"
else
    echo "   .env file NOT FOUND (copy from .env.example)"
fi
echo ""

echo "=== Smoke Test Complete ==="
echo ""
echo "To run full experiments:"
echo "  bash scripts/eval_llada_gsm8k.sh"
echo "  bash scripts/eval_dream.sh"
echo "  bash scripts/eval_qwen_gsm8k.sh"
