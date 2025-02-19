# Quantitative Strategies Documentation

## Overview

The Quantitative Strategies module (`backend/app/strategies/quantitative_strategies.py`) implements high-frequency trading (HFT), market making, and statistical arbitrage strategies optimized for low initial capital and minimal overhead costs.

## Components

### 1. Configuration Parameters

```python
@dataclass
class QuantParams:
    min_spread: float = 0.001          # Minimum spread for trading (0.1%)
    max_position_hold_time: int = 300   # Maximum position hold time (5 minutes)
    min_volume_threshold: float = 100   # Minimum volume requirement
    correlation_threshold: float = 0.7   # Correlation threshold for pair trading
    mean_reversion_threshold: float = 2.0  # Z-score threshold
    market_making_depth: int = 3        # Order book depth levels
    hft_timeout: float = 0.5           # Maximum HFT position hold time
```

### 2. High-Frequency Trading (HFT)

#### Strategy Overview
- Ultra-low latency execution
- Order book imbalance detection
- Dynamic spread capture
- Minimal position hold time (0.5s)

#### Implementation Details
```python
async def execute_hft_strategy(
    self, 
    token_pair: str,
    order_book: Dict,
    market_data: Dict
) -> Optional[Dict]
```

#### Key Features
- **Order Book Analysis**
  - Calculates bid-ask imbalances
  - Detects price pressure
  - Monitors liquidity levels

- **Position Sizing**
  - Dynamic sizing based on imbalance
  - Volume-adjusted positions
  - Risk-aware allocation

- **Execution Logic**
  - Sub-second execution
  - Smart order routing
  - Slippage optimization

### 3. Market Making

#### Strategy Overview
- Dynamic spread adjustment
- Multi-level order grid
- Volatility-based pricing
- Risk-adjusted position sizes

#### Implementation Details
```python
async def execute_market_making(
    self,
    token_pair: str,
    order_book: Dict,
    market_data: Dict
) -> List[Dict]
```

#### Key Features
- **Spread Management**
  - Dynamic spread calculation
  - Volatility adjustment
  - Market depth consideration

- **Order Grid**
  - Multi-level placement
  - Volume-weighted spacing
  - Risk-adjusted sizes

- **Risk Controls**
  - Maximum exposure limits
  - Dynamic rebalancing
  - Inventory management

### 4. Statistical Arbitrage

#### Strategy Overview
- Correlation-based pair detection
- Mean reversion trading
- Z-score based signals
- Cointegration analysis

#### Implementation Details
```python
async def execute_stat_arb(
    self,
    token_pairs: List[str],
    price_data: Dict[str, pd.DataFrame]
) -> Optional[Dict]
```

#### Key Features
- **Pair Selection**
  - Correlation analysis
  - Cointegration testing
  - Liquidity filtering

- **Signal Generation**
  - Z-score calculation
  - Mean reversion detection
  - Trend filtering

- **Position Management**
  - Pair-balanced positions
  - Dynamic hedge ratios
  - Risk-adjusted sizing

## Position Sizing and Risk Management

### 1. HFT Position Sizing
```python
def _calculate_hft_position_size(
    self,
    price: float,
    spread: float,
    volume: float
) -> float
```
- Uses 1% of available capital
- Adjusts for spread width
- Scales with volume

### 2. Market Making Position Sizing
```python
def _calculate_mm_position_size(
    self,
    price: float,
    spread: float,
    level: int
) -> float
```
- Base size decreases with depth
- Spread-adjusted sizing
- Level-based reduction

### 3. Statistical Arbitrage Sizing
```python
def _calculate_stat_arb_position_size(
    self,
    opportunity: Dict
) -> float
```
- Uses 5% of capital
- Correlation-adjusted
- Balanced pair exposure

## Performance Optimization

### 1. Latency Optimization
- Asynchronous execution
- Minimal computation
- Efficient data structures

### 2. Resource Management
- Memory-efficient calculations
- Batch processing
- Cache optimization

### 3. Cost Efficiency
- Minimal position sizes
- Spread requirement enforcement
- Volume-based filtering

## Usage Examples

### 1. HFT Strategy
```python
# Execute HFT strategy
hft_signal = await quant_strategies.execute_hft_strategy(
    token_pair="SOL/USDC",
    order_book=current_order_book,
    market_data=market_features
)

if hft_signal:
    await execute_trade(
        token_pair=hft_signal['pair'],
        amount=hft_signal['size'],
        trade_type=hft_signal['action']
    )
```

### 2. Market Making
```python
# Execute market making
mm_orders = await quant_strategies.execute_market_making(
    token_pair="SOL/USDC",
    order_book=current_order_book,
    market_data=market_features
)

for order in mm_orders:
    await execute_trade(
        token_pair=order['pair'],
        amount=order['size'],
        trade_type=order['action']
    )
```

### 3. Statistical Arbitrage
```python
# Execute statistical arbitrage
stat_arb = await quant_strategies.execute_stat_arb(
    token_pairs=valid_pairs,
    price_data=price_cache
)

if stat_arb:
    await execute_trade(
        token_pair=stat_arb['pair_a'],
        amount=stat_arb['size_a'],
        trade_type=stat_arb['action_a']
    )
```

## Error Handling

### 1. Volume Validation
```python
if market_data.get('volume', 0) < self.params.min_volume_threshold:
    return None
```

### 2. Spread Validation
```python
if spread < self.params.min_spread:
    return None
```

### 3. Position Timeout
```python
if datetime.now() > position['timeout']:
    await close_position(position)
```

## Monitoring and Metrics

### 1. Strategy Performance
- Win/loss ratio
- Average profit per trade
- Strategy-specific metrics

### 2. Resource Usage
- Execution latency
- Memory utilization
- API call frequency

### 3. Risk Metrics
- Position exposure
- Correlation metrics
- Drawdown tracking 