import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class QuantParams:
    min_spread: float = 0.001  # 0.1% minimum spread
    max_position_hold_time: int = 300  # 5 minutes
    min_volume_threshold: float = 100  # Minimum volume for liquidity
    correlation_threshold: float = 0.7
    mean_reversion_threshold: float = 2.0  # Z-score threshold
    market_making_depth: int = 3  # Order book depth
    hft_timeout: float = 0.5  # Maximum time to hold HFT position

class QuantitativeStrategies:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.params = QuantParams()
        self.position_cache: Dict[str, Dict] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.market_making_positions: Dict[str, List[Dict]] = {}
        
    async def execute_hft_strategy(self, 
                                 token_pair: str,
                                 order_book: Dict,
                                 market_data: Dict) -> Optional[Dict]:
        """Execute HFT strategy with minimal latency"""
        # Skip if insufficient liquidity
        if market_data.get('volume', 0) < self.params.min_volume_threshold:
            return None
            
        # Calculate bid-ask spread
        best_bid = max(level['price'] for level in order_book['bids'])
        best_ask = min(level['price'] for level in order_book['asks'])
        spread = (best_ask - best_bid) / best_bid
        
        if spread < self.params.min_spread:
            return None
            
        # Detect order book imbalance
        imbalance = self._calculate_order_book_imbalance(order_book)
        
        # Execute if significant imbalance detected
        if abs(imbalance) > 0.2:  # 20% imbalance threshold
            position_size = self._calculate_hft_position_size(
                market_data['price'],
                spread,
                market_data['volume']
            )
            
            return {
                'action': 'buy' if imbalance > 0 else 'sell',
                'size': position_size,
                'price': best_bid if imbalance > 0 else best_ask,
                'type': 'hft',
                'timeout': datetime.now() + timedelta(seconds=self.params.hft_timeout)
            }
        
        return None
        
    async def execute_market_making(self,
                                  token_pair: str,
                                  order_book: Dict,
                                  market_data: Dict) -> List[Dict]:
        """Market making with dynamic spread adjustment"""
        if market_data.get('volume', 0) < self.params.min_volume_threshold:
            return []
            
        volatility = market_data.get('volatility', 0.1)
        mid_price = (order_book['bids'][0]['price'] + order_book['asks'][0]['price']) / 2
        
        # Calculate dynamic spread based on volatility
        dynamic_spread = max(
            self.params.min_spread,
            volatility * 0.5  # Spread increases with volatility
        )
        
        # Generate order grid
        orders = []
        for i in range(self.params.market_making_depth):
            spread_multiplier = 1 + (i * 0.5)  # Increase spread for each level
            position_size = self._calculate_mm_position_size(
                market_data['price'],
                dynamic_spread * spread_multiplier,
                i
            )
            
            orders.extend([
                {
                    'action': 'buy',
                    'price': mid_price * (1 - dynamic_spread * spread_multiplier),
                    'size': position_size,
                    'type': 'maker'
                },
                {
                    'action': 'sell',
                    'price': mid_price * (1 + dynamic_spread * spread_multiplier),
                    'size': position_size,
                    'type': 'maker'
                }
            ])
            
        return orders
        
    async def execute_stat_arb(self,
                              token_pairs: List[str],
                              price_data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Statistical arbitrage with minimal computational overhead"""
        # Update correlation matrix periodically
        if self._should_update_correlations():
            self._update_correlation_matrix(price_data)
            
        # Find highly correlated pairs
        arb_opportunities = self._find_cointegrated_pairs(price_data)
        
        if not arb_opportunities:
            return None
            
        # Select best opportunity
        best_opportunity = max(
            arb_opportunities,
            key=lambda x: abs(x['zscore'])
        )
        
        if abs(best_opportunity['zscore']) > self.params.mean_reversion_threshold:
            return {
                'pair_a': best_opportunity['pair_a'],
                'pair_b': best_opportunity['pair_b'],
                'action_a': 'buy' if best_opportunity['zscore'] < 0 else 'sell',
                'action_b': 'sell' if best_opportunity['zscore'] < 0 else 'buy',
                'size_a': self._calculate_stat_arb_position_size(best_opportunity),
                'type': 'stat_arb'
            }
            
        return None
        
    def _calculate_order_book_imbalance(self, order_book: Dict) -> float:
        """Calculate order book imbalance ratio"""
        bid_volume = sum(level['size'] for level in order_book['bids'][:5])
        ask_volume = sum(level['size'] for level in order_book['asks'][:5])
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
    def _calculate_hft_position_size(self,
                                   price: float,
                                   spread: float,
                                   volume: float) -> float:
        """Calculate optimal HFT position size"""
        # Use 1% of available capital, adjusted for spread and volume
        base_size = self.initial_capital * 0.01
        # Adjust size based on spread (larger spread = larger position)
        spread_factor = min(spread / self.params.min_spread, 2.0)
        # Adjust for volume (lower volume = smaller position)
        volume_factor = min(volume / self.params.min_volume_threshold, 1.0)
        
        return base_size * spread_factor * volume_factor
        
    def _calculate_mm_position_size(self,
                                  price: float,
                                  spread: float,
                                  level: int) -> float:
        """Calculate market making position size per level"""
        # Base size decreases as we go deeper in the order book
        base_size = self.initial_capital * 0.02 * (1 / (level + 1))
        # Adjust based on spread
        spread_factor = min(spread / self.params.min_spread, 2.0)
        
        return base_size * spread_factor
        
    def _calculate_stat_arb_position_size(self, opportunity: Dict) -> float:
        """Calculate position size for statistical arbitrage"""
        # Use 5% of capital for stat arb, adjusted for correlation strength
        base_size = self.initial_capital * 0.05
        correlation_factor = min(
            abs(opportunity['correlation']),
            self.params.correlation_threshold
        ) / self.params.correlation_threshold
        
        return base_size * correlation_factor
        
    def _should_update_correlations(self) -> bool:
        """Check if correlation matrix needs updating"""
        if not hasattr(self, '_last_correlation_update'):
            self._last_correlation_update = datetime.now()
            return True
            
        return (datetime.now() - self._last_correlation_update) > timedelta(hours=1)
        
    def _update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix for pairs"""
        prices = pd.DataFrame({
            pair: df['price'].values
            for pair, df in price_data.items()
        })
        
        self.correlation_matrix = prices.corr()
        self._last_correlation_update = datetime.now()
        
    def _find_cointegrated_pairs(self,
                               price_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find cointegrated pairs for statistical arbitrage"""
        opportunities = []
        pairs = list(price_data.keys())
        
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                pair_a, pair_b = pairs[i], pairs[j]
                correlation = self.correlation_matrix.loc[pair_a, pair_b]
                
                if abs(correlation) > self.params.correlation_threshold:
                    # Calculate z-score of price ratio
                    ratio = (price_data[pair_a]['price'] / 
                            price_data[pair_b]['price'])
                    zscore = (ratio - ratio.mean()) / ratio.std()
                    
                    opportunities.append({
                        'pair_a': pair_a,
                        'pair_b': pair_b,
                        'correlation': correlation,
                        'zscore': zscore.iloc[-1]
                    })
                    
        return opportunities 