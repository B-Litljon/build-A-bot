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
    echo "Conservative execution: Stop Loss checked before Take Profit"
    echo ""
    
    python -m src.core.resolver
    
    print_success "Trade resolution complete"
    echo "Output: data/resolved_ledger.csv"
}

# Phase 4: Drift Evaluation
run_feedback() {
    print_section "PHASE 4: Drift Evaluation"
    echo "Evaluating model performance metrics..."
    echo "Metrics: Win Rate, Expected Value, Brier Score, Log Loss"
    echo "Thresholds: Brier <= 0.25, EV >= 0.0005 (0.05%)"
    echo ""
    
    python -m src.core.feedback_loop
    
    # Note: feedback_loop returns non-zero exit code if drift detected
    # set -e handles this, but we want to distinguish between errors and drift alerts
}

# Handle pipeline completion
handle_completion() {
    local exit_code=$?
    
    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "OOS PIPELINE COMPLETED SUCCESSFULLY"
        echo ""
        echo "Generated files:"
        echo "  - data/oos_bars.parquet (historical data)"
        echo "  - data/signal_ledger.csv (trade signals)"
        echo "  - data/resolved_ledger.csv (resolved outcomes)"
        echo ""
        print_success "Model health: WITHIN PARAMETERS"
    elif [[ $exit_code -eq 1 ]]; then
        # This could be an error or drift detection
        # The feedback_loop should have printed appropriate messages
        print_warning "Pipeline completed with warnings or drift detected"
        print_warning "Check logs above for details"
    else
        print_error "Pipeline failed with exit code $exit_code"
        exit $exit_code
    fi
    
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
}

# Handle interrupts
trap 'print_error "Pipeline interrupted by user"; exit 130' INT TERM

# Main execution
main() {
    local start_time=$(date +%s)
    
    print_header
    
    # Pre-flight checks
    check_environment
    check_python
    
    # Run pipeline phases
    run_harvester
    run_replay
    run_resolver
    run_feedback
    
    # Calculate execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo ""
    print_success "Total execution time: ${minutes}m ${seconds}s"
    
    handle_completion
}

# Run main function
main "$@"
