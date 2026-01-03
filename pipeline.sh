#!/bin/bash
#
# Pipeline script for parallelism benchmark
# Runs training and multi-trial evaluation
#
# Usage:
#   ./pipeline.sh              # Run with default 100 trials
#   ./pipeline.sh 50           # Run with 50 trials
#   ./pipeline.sh 100 --skip-train  # Skip training, only run evaluation
#

set -e  # Exit on error

# Configuration
NUM_TRIALS=${1:-100}
SKIP_TRAIN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Parallelism Benchmark Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check Python environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Create output directory
mkdir -p saved_artifacts

# Step 1: Train models (generates Silver Standard + training splits)
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "${GREEN}Step 1: Training models...${NC}"
    echo "  This will also generate:"
    echo "    - data/silver_standard.json (Silver Standard dataset)"
    echo "    - saved_artifacts/*.json (training/test splits)"
    echo ""
    python3 train_models.py
    echo ""
    echo -e "${GREEN}Training complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping training (--skip-train flag set)${NC}"
    echo ""
fi

# Step 2: Run multi-trial evaluation
echo -e "${GREEN}Step 2: Running ${NUM_TRIALS}-trial evaluation...${NC}"
echo "  This trains and evaluates models ${NUM_TRIALS} times with different seeds"
echo "  to compute mean ± std statistics."
echo ""
python3 evaluate.py --trials ${NUM_TRIALS} --output evaluation_results.json
echo ""

# Step 3: Display results summary
echo -e "${GREEN}Step 3: Results Summary${NC}"
echo -e "${BLUE}============================================${NC}"
python3 -c "
import json
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

print(f\"Number of trials: {data['num_trials']}\")
print()
print('Metric                      Mean ± Std')
print('-' * 45)
for key, stats in data['statistics'].items():
    mean = stats['mean']
    std = stats['std']
    print(f'{key:28} {mean:.4f} ± {std:.4f}')
"
echo -e "${BLUE}============================================${NC}"
echo ""

# Step 4: List generated files
echo -e "${GREEN}Generated files:${NC}"
echo "  - data/silver_standard.json"
echo "  - saved_artifacts/char_train.json, char_test.json"
echo "  - saved_artifacts/coup_train.json, coup_test.json"
echo "  - saved_artifacts/poem4_train.json, poem4_test.json"
echo "  - saved_artifacts/poem1_train.json, poem1_test.json"
echo "  - saved_artifacts/{char,coup,poem4,poem1}_model/"
echo "  - evaluation_results.json"
echo ""

echo -e "${GREEN}Pipeline complete!${NC}"

