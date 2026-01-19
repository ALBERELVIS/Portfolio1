# Portfolio1

## Asset search tool

Use `asset_search.py` to scan the training dataset and highlight assets with unusual or desirable behavior (high mean, low volatility, strong Sharpe-like ratios, etc.). The tool reads the provided CSV and ranks assets so you can quickly spot the "mysterious" standouts.

### Quick start

```bash
python asset_search.py --top 10
```

### Examples

Find assets with strong average returns and low volatility:

```bash
python asset_search.py --min-mean 0.002 --max-volatility 0.01 --sort sharpe --top 5
```

Filter by name pattern and sort by downside tail risk (CVaR):

```bash
python asset_search.py --name-regex "asset(1|2|3)" --sort cvar_95
```

### Metrics included

- **mean**: average return per period
- **volatility**: standard deviation of returns
- **sharpe**: mean divided by volatility (risk-adjusted)
- **% positive**: percentage of positive observations
- **cvar_95**: average of worst 5% returns (tail risk)
- **range**: max minus min return (for sorting)
