#!/bin/bash
#
# Pipeline script for parallelism benchmark
#
# Usage:
#   ./scripts/pipeline.sh              # Prepare data + run 1 trial
#   ./scripts/pipeline.sh 100          # Prepare data + run 100 trials
#   ./scripts/pipeline.sh --skip-prep  # Skip data preparation, only run trials
#   ./scripts/pipeline.sh 50 --skip-prep  # Run 50 trials without data prep
#   ./scripts/pipeline.sh --max-poems 10000  # Only classify 10k poems (faster)
#

set -e  # Exit on error

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Configuration
NUM_TRIALS=1
SKIP_PREP=false
MAX_POEMS=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-prep)
            SKIP_PREP=true
            ;;
        --max-poems)
            # Next arg will be the value
            ;;
        [0-9]*)
            # Check if previous arg was --max-poems
            if [[ "${PREV_ARG}" == "--max-poems" ]]; then
                MAX_POEMS=$arg
            else
                NUM_TRIALS=$arg
            fi
            ;;
    esac
    PREV_ARG=$arg
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
    echo "  This classifies couplets with SikuBERT (slow, run once)"
    echo ""
    
    PREP_CMD="python3 scripts/prepare_data.py"
    if [ -n "$MAX_POEMS" ]; then
        PREP_CMD="$PREP_CMD --max-poems $MAX_POEMS"
        echo "  (Limiting to $MAX_POEMS poems)"
    fi
    $PREP_CMD
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
python3 scripts/run_trials.py --trials ${NUM_TRIALS}
echo ""

# Step 3: Display results summary
echo -e "${GREEN}Results Summary${NC}"
echo -e "${BLUE}============================================${NC}"
python3 -c "
import json
with open('results/evaluation_results.json', 'r') as f:
    data = json.load(f)

print(f\"Number of trials: {data['num_trials']}\")
print()

if 'statistics' in data:
    print('Metric                              Mean ± Std')
    print('-' * 55)
    for key, stats in data['statistics'].items():
        mean = stats['mean']
        std = stats['std']
        print(f'{key:35} {mean:.4f} ± {std:.4f}')
else:
    print('Single trial results:')
    print('-' * 55)
    trial = data['trials'][0]
    for key, value in trial.items():
        if key == 'seed':
            continue
        if isinstance(value, dict) and 'accuracy' in value:
            print(f'{key:35} {value[\"accuracy\"]:.4f}')
"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${GREEN}Generated files:${NC}"
echo "  - data/silver_standard.json (pre-classified poems)"
echo "  - results/evaluation_results.json (trial results)"
echo ""

echo -e "${GREEN}Pipeline complete!${NC}"
