#!/bin/bash
#
# Master Orchestrator Script - Universal Scalper v3.0
# Out-of-Sample (OOS) Testing and Reinforcement Feedback Loop
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Description:
#   Automates the complete OOS testing pipeline:
#   1. Harvests 7 days of 1-minute bars from Alpaca
#   2. Runs replay simulation with Angel/Devil models
#   3. Resolves trades to Win/Loss outcomes
#   4. Evaluates model drift and sends alerts if needed
#

# Safety: Exit immediately if any command fails
set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_NAME="run_pipeline.sh"
ENV_FILE=".env"

# Helper functions
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           UNIVERSAL SCALPER v3.0 - OOS PIPELINE                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BLUE}[+] $1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..70})${NC}"
}

print_success() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

# Check if .env file exists and load it
check_environment() {
    print_section "Loading Environment"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error "Environment file not found: $ENV_FILE"
        print_error "Please create $ENV_FILE with the following variables:"
        echo "  ALPACA_API_KEY=your_api_key"
        echo "  ALPACA_SECRET_KEY=your_secret_key"
        exit 1
    fi
    
    # Source the .env file
    set -a
    source "$ENV_FILE"
    set +a
    
    print_success "Loaded environment from $ENV_FILE"
    
    # Verify required variables
    local missing_vars=()
    
    if [[ -z "$ALPACA_API_KEY" ]]; then
        missing_vars+=("ALPACA_API_KEY")
    fi
    
    if [[ -z "$ALPACA_SECRET_KEY" ]]; then
        missing_vars+=("ALPACA_SECRET_KEY")
    fi
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
    
    print_success "Environment variables validated"
}

# Check Python environment
check_python() {
    print_section "Checking Python Environment"
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
    print_success "Python version: $PYTHON_VERSION"
}

# Phase 1: Data Harvesting
run_harvester() {
    print_section "PHASE 1: Data Harvesting"
    echo "Fetching 7 days of 1-minute bars from Alpaca API..."
    echo "Tickers: TSLA, NVDA, MARA, COIN, SMCI"
    echo ""
    
    python -m src.data.harvester
    
    print_success "Data harvesting complete"
    echo "Output: data/oos_bars.parquet"
}

# Phase 2: Replay Simulation
run_replay() {
    print_section "PHASE 2: Replay Simulation"
    echo "Running Angel/Devil model inference on OOS data..."
    echo "Loading models:"
    echo "  - Angel RF (Recall) -> Entry threshold: 0.40"
    echo "  - Devil RF (Precision) -> Approval threshold: 0.50"
    echo ""
    
    python -m src.replay_test
    
    print_success "Replay simulation complete"
    echo "Output: data/signal_ledger.csv"
}

# Phase 3: Trade Resolution
run_resolver() {
    print_section "PHASE 3: Trade Resolution"
    echo "Resolving bracket orders (+0.5% TP, -0.2% SL)..."
    echo "Dynamic ATR-based TP/SL evaluation with conservative execution"
    echo ""
    
    python -m src.evaluate_performance
    
    print_success "Trade resolution complete"
    echo "Output: data/resolved_ledger.csv"
}

# Phase 4: Drift Evaluation
# Exit codes: 0 = Healthy, 1 = Error, 2 = Critical Drift (triggers retraining)
run_feedback() {
    print_section "PHASE 4: Drift Evaluation"
    echo "Evaluating model performance metrics..."
    echo "Metrics: Win Rate, Expected Value, Brier Score, Log Loss"
    echo "Thresholds: Brier <= 0.25, EV >= 0.0005 (0.05%)"
    echo ""
    
    # Temporarily disable exit-on-error to capture feedback_loop exit code
    set +e
    python -m src.core.feedback_loop
    FEEDBACK_STATUS=$?
    set -e
    
    return $FEEDBACK_STATUS
}

# Phase 5: Model Retraining (The Cure)
# Exit codes from retrainer:
#   0 = models promoted (gate passed, full-data model saved)
#   1 = execution error (data fetch failed, config error, etc.)
#   2 = models rejected by validation gate (production weights retained)
run_retrainer() {
    print_section "PHASE 5: THE CURE - Validated Model Retraining"
    echo "Critical drift detected. Initiating walk-forward validated retraining..."
    echo "Fetching 60 days of fresh training data..."
    echo "Running 3-fold expanding-window cross-validation gate..."
    echo ""

    # Disable exit-on-error so exit code 2 (gate rejected) doesn't abort the script
    set +e
    python -m src.core.retrainer
    RETRAINER_STATUS=$?
    set -e

    return $RETRAINER_STATUS
}

# Handle pipeline completion based on feedback and retrainer status
# Args:
#   $1 = feedback_status  (0=healthy, 2=drift triggered retraining)
#   $2 = retrainer_status (0=promoted, 1=error, 2=rejected) — only set when $1==2
handle_completion() {
    local feedback_status=$1
    local retrainer_status=${2:-0}

    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"

    if [[ $feedback_status -eq 0 ]]; then
        # Healthy - no retraining needed
        print_success "OOS PIPELINE COMPLETED SUCCESSFULLY"
        echo ""
        echo "Generated files:"
        echo "  - data/oos_bars.parquet (historical data)"
        echo "  - data/signal_ledger.csv (trade signals)"
        echo "  - data/resolved_ledger.csv (resolved outcomes)"
        echo ""
        print_success "Model health: WITHIN PARAMETERS"

    elif [[ $feedback_status -eq 2 ]]; then
        # Drift was detected and retraining was attempted — branch on retrainer result
        if [[ $retrainer_status -eq 0 ]]; then
            # Gate passed — new models promoted
            print_success "CRITICAL DRIFT DETECTED — RETRAINING COMPLETE — MODELS PROMOTED"
            echo ""
            echo "New models passed walk-forward validation and are now live:"
            echo "  - models/angel_latest.pkl  (trained on 60 days, full-data)"
            echo "  - models/devil_latest.pkl  (trained on 60 days, full-data)"
            echo ""
            print_success "AUTONOMOUS RECOVERY COMPLETE"

        elif [[ $retrainer_status -eq 2 ]]; then
            # Gate rejected — production weights retained
            print_warning "⚠️  CRITICAL DRIFT DETECTED — RETRAINING ATTEMPTED — MODELS REJECTED"
            echo ""
            echo "Candidate models failed walk-forward validation gate."
            echo "Production weights were NOT replaced. Manual review recommended."
            echo ""
            echo "Check the retrainer log for rejection reasons:"
            echo "  Brier Score  > 0.25  OR"
            echo "  Expected Value < 0.0005  OR"
            echo "  Profit Factor (Fold 3 OOS) < 1.2"
            echo ""
            print_warning "MANUAL REVIEW REQUIRED"

        else
            # Retrainer crashed (exit code 1)
            print_error "RETRAINING FAILED WITH EXECUTION ERROR (exit code $retrainer_status)"
            echo ""
            echo "The retrainer encountered an unexpected error."
            echo "Production weights were NOT replaced."
            exit 1
        fi

    else
        # Feedback loop itself errored
        print_error "Pipeline failed with exit code $feedback_status"
        exit 1
    fi

    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
}

# Handle interrupts
trap 'print_error "Pipeline interrupted by user"; exit 130' INT TERM

# Main execution
main() {
    local start_time=$(date +%s)
    local feedback_status=0
    local retrainer_status=0

    print_header

    # Pre-flight checks
    check_environment
    check_python

    # Run pipeline phases 1-3 (these must succeed)
    run_harvester
    run_replay
    run_resolver

    # Phase 4: Drift evaluation (handles its own exit codes)
    run_feedback
    feedback_status=$?

    # Route based on feedback status
    if [[ $feedback_status -eq 2 ]]; then
        # Critical drift detected - trigger retraining
        echo ""
        echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║              🚨 CRITICAL DRIFT ALERT 🚨                          ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "[+] INITIATING THE CURE (walk-forward validated)..."
        echo ""

        run_retrainer
        retrainer_status=$?
    fi

    # Calculate execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo ""
    print_success "Total execution time: ${minutes}m ${seconds}s"

    # Handle completion based on both feedback and retrainer status
    handle_completion $feedback_status $retrainer_status
}

# Run main function
main "$@"
