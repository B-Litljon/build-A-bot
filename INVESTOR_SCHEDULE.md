# Investor Rebalance Scheduling Instructions

This document provides instructions on how to set up the cron schedule for the monthly Equities Investor rebalance.

## Scheduling with Cron

To schedule the investor monthly rebalance, edit your crontab using:

```bash
crontab -e
```

And add the following line:

```text
30 16 1 * *  cd /mnt/storage/mystuf/development/build-A-bot && ./run_investor_rebalance.sh --dry-run
```

### Important Notes:
1. **Initially Keep `--dry-run`**: The first scheduled runs should stay configured with `--dry-run` to log intended trades without sending them. This allows review of the intended allocations.
2. **Going Live**: Once you are comfortable with the generated allocations and want to execute trades live on the Alpaca paper account, drop the `--dry-run` flag from the cron command.
