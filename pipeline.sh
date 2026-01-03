#!/bin/bash
#
# Pipeline script for parallelism benchmark
#
# Usage:
#   ./pipeline.sh              # Prepare data + run 1 trial
#   ./pipeline.sh 100          # Prepare data + run 100 trials
#   ./pipeline.sh --skip-prep  # Skip data preparation, only run trials
#   ./pipeline.sh 50 --skip-prep  # Run 50 trials without data prep
#

set -e  # Exit on error

# Configuration
NUM_TRIALS=${1:-1}
SKIP_PREP=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-prep)
            SKIP_PREP=true
            ;;
        [0-9]*)
            NUM_TRIALS=$arg
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

# Step 1: Prepare data (expensive, run once)
if [ "$SKIP_PREP" = false ]; then
    echo -e "${GREEN}Step 1: Preparing data...${NC}"
    echo "  This classifies all couplets with SikuBERT (slow, run once)"
    echo ""
    python3 prepare_data.py
    echo ""
else
    if [ ! -f "data/silver_standard.json" ]; then
        echo -e "${YELLOW}Warning: data/silver_standard.json not found!${NC}"
        echo "  Run without --skip-prep first to generate the data."
        exit 1
    fi
    echo -e "${YELLOW}Skipping data preparation (--skip-prep flag set)${NC}"
    echo ""
fi

# Step 2: Run trials
echo -e "${GREEN}Step 2: Running ${NUM_TRIALS} trial(s)...${NC}"
echo "  Each trial samples data, trains models, and evaluates."
echo ""
python3 run_trials.py --trials ${NUM_TRIALS} --output evaluation_results.json
echo ""

# Step 3: Display results summary
echo -e "${GREEN}Results Summary${NC}"
echo -e "${BLUE}============================================${NC}"
python3 -c "
import json
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

print(f\"Number of trials: {data['num_trials']}\")
print()

if 'statistics' in data:
    print('Metric                      Mean ± Std')
    print('-' * 45)
    for key, stats in data['statistics'].items():
        mean = stats['mean']
        std = stats['std']
        print(f'{key:28} {mean:.4f} ± {std:.4f}')
else:
    print('Single trial results:')
    print('-' * 45)
    for key, value in data['trials'][0].items():
        if key != 'seed':
            print(f'{key:28} {value:.4f}')
"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${GREEN}Generated files:${NC}"
echo "  - data/silver_standard.json (pre-classified poems)"
echo "  - evaluation_results.json (trial results)"
echo ""

echo -e "${GREEN}Pipeline complete!${NC}"
