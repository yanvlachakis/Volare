import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    max_position_size: float
    max_drawdown: float
    max_daily_loss: float
    max_leverage: float
    min_profit_target: float
    trailing_stop_distance: float
    stop_loss_distance: float
    risk_per_trade: float
    max_correlation: float
    volatility_threshold: float

class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.open_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.risk_metrics: Dict[str, float] = {}
        
        # Load risk parameters from environment or config
        self.risk_params = RiskParameters(
            max_position_size=0.1,  # 10% of portfolio
            max_drawdown=0.15,      # 15% max drawdown
            max_daily_loss=0.05,    # 5% max daily loss
            max_leverage=2.0,       # 2x max leverage
            min_profit_target=0.01,  # 1% min profit target
            trailing_stop_distance=0.02,  # 2% trailing stop
            stop_loss_distance=0.03,     # 3% stop loss
            risk_per_trade=0.01,         # 1% risk per trade
            max_correlation=0.7,         # 70% max correlation
            volatility_threshold=0.5     # 50% volatility threshold
        )
        
    def calculate_position_size(self,
                              token_pair: str,
                              entry_price: float,
                              volatility: float,
                              market_impact: float) -> Tuple[float, Dict]:
        """Calculates safe position size based on multiple risk factors"""
        # Get current portfolio state
        portfolio_risk = self._calculate_portfolio_risk()
        available_risk = max(0, self.risk_params.risk_per_trade - portfolio_risk)
        
        # Calculate position size based on Kelly Criterion with safety factor
        win_rate = self._get_historical_win_rate(token_pair)
        profit_ratio = self._get_historical_profit_ratio(token_pair)
        kelly_fraction = self._calculate_kelly_fraction(win_rate, profit_ratio)
        
        # Apply volatility adjustment
        volatility_factor = np.exp(-volatility / self.risk_params.volatility_threshold)
        
        # Consider market impact
        impact_factor = np.exp(-market_impact * 10)  # Exponential decay for high impact
        
        # Calculate base position size
        max_position = self.current_capital * self.risk_params.max_position_size
        position_size = max_position * kelly_fraction * volatility_factor * impact_factor
        
        # Apply additional risk constraints
        position_size = min(
            position_size,
            self._get_max_position_size_by_drawdown(),
            self._get_max_position_size_by_daily_loss(),
            self._get_max_position_size_by_leverage()
        )
        
        risk_metrics = {
            'kelly_fraction': kelly_fraction,
            'volatility_factor': volatility_factor,
            'impact_factor': impact_factor,
            'portfolio_risk': portfolio_risk,
            'available_risk': available_risk
        }
        
        return position_size, risk_metrics
        
    def manage_open_position(self,
                           token_pair: str,
                           current_price: float,
                           position_data: Dict) -> Dict:
        """Manages open position with dynamic stop loss and take profit"""
        if token_pair not in self.open_positions:
            return {'action': 'none'}
            
        position = self.open_positions[token_pair]
        entry_price = position['entry_price']
        position_size = position['size']
        
        # Calculate current P&L
        unrealized_pnl = self._calculate_unrealized_pnl(
            position_size, entry_price, current_price, position['side']
        )
        
        # Update trailing stop if profitable
        if unrealized_pnl > 0:
            self._update_trailing_stop(token_pair, current_price, position['side'])
        
        # Check stop conditions
        stop_triggered = self._check_stop_conditions(
            token_pair, current_price, position['side']
        )
        
        if stop_triggered:
            return {
                'action': 'close',
                'reason': stop_triggered['reason'],
                'pnl': unrealized_pnl
            }
            
        # Check take profit conditions
        if self._check_take_profit_conditions(token_pair, unrealized_pnl):
            return {
                'action': 'reduce',
                'size_reduction': position_size * 0.5,  # Reduce position by 50%
                'pnl': unrealized_pnl
            }
            
        return {
            'action': 'hold',
            'current_pnl': unrealized_pnl,
            'risk_metrics': self._calculate_position_risk_metrics(token_pair)
        }
        
    def update_position(self,
                       token_pair: str,
                       trade_result: Dict):
        """Updates position tracking with new trade results"""
        if trade_result['action'] == 'open':
            self.open_positions[token_pair] = {
                'entry_price': trade_result['price'],
                'size': trade_result['size'],
                'side': trade_result['side'],
                'timestamp': datetime.now(),
                'initial_stop': trade_result['price'] * (
                    1 - self.risk_params.stop_loss_distance
                    if trade_result['side'] == 'buy'
                    else 1 + self.risk_params.stop_loss_distance
                ),
                'trailing_stop': None
            }
        elif trade_result['action'] in ['close', 'reduce']:
            if token_pair in self.open_positions:
                position = self.open_positions[token_pair]
                pnl = trade_result.get('pnl', 0)
                
                # Update capital
                self.current_capital += pnl
                self.peak_capital = max(self.peak_capital, self.current_capital)
                
                # Update daily P&L
                today = datetime.now().date().isoformat()
                self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
                
                # Record position history
                self.position_history.append({
                    'token_pair': token_pair,
                    'entry_price': position['entry_price'],
                    'exit_price': trade_result['price'],
                    'size': position['size'],
                    'pnl': pnl,
                    'duration': (datetime.now() - position['timestamp']).total_seconds(),
                    'side': position['side']
                })
                
                if trade_result['action'] == 'close':
                    del self.open_positions[token_pair]
                else:
                    self.open_positions[token_pair]['size'] -= trade_result['size_reduction']
                    
    def _calculate_portfolio_risk(self) -> float:
        """Calculates current portfolio risk level"""
        if not self.open_positions:
            return 0.0
            
        position_risks = []
        for token_pair, position in self.open_positions.items():
            volatility = self._get_token_volatility(token_pair)
            position_size_ratio = position['size'] / self.current_capital
            position_risks.append(position_size_ratio * volatility)
            
        # Consider position correlation
        correlation_matrix = self._calculate_position_correlation()
        portfolio_risk = np.sqrt(
            np.sum(correlation_matrix * np.outer(position_risks, position_risks))
        )
        
        return portfolio_risk
        
    def _get_historical_win_rate(self, token_pair: str) -> float:
        """Calculates historical win rate for a token pair"""
        relevant_history = [
            trade for trade in self.position_history
            if trade['token_pair'] == token_pair
        ]
        
        if not relevant_history:
            return 0.5  # Default to 50% if no history
            
        winning_trades = sum(1 for trade in relevant_history if trade['pnl'] > 0)
        return winning_trades / len(relevant_history)
        
    def _get_historical_profit_ratio(self, token_pair: str) -> float:
        """Calculates historical profit ratio for a token pair"""
        relevant_history = [
            trade for trade in self.position_history
            if trade['token_pair'] == token_pair
        ]
        
        if not relevant_history:
            return 1.5  # Default to 1.5 if no history
            
        avg_profit = np.mean([t['pnl'] for t in relevant_history if t['pnl'] > 0] or [0])
        avg_loss = abs(np.mean([t['pnl'] for t in relevant_history if t['pnl'] < 0] or [0]))
        
        return avg_profit / avg_loss if avg_loss > 0 else 1.5
        
    def _calculate_kelly_fraction(self,
                                win_rate: float,
                                profit_ratio: float) -> float:
        """Calculates Kelly Criterion fraction with safety factor"""
        kelly = win_rate - (1 - win_rate) / profit_ratio
        return max(0, kelly * 0.5)  # Apply 50% safety factor
        
    def _get_max_position_size_by_drawdown(self) -> float:
        """Calculates maximum position size based on drawdown limit"""
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        remaining_drawdown = max(0, self.risk_params.max_drawdown - current_drawdown)
        return self.current_capital * remaining_drawdown
        
    def _get_max_position_size_by_daily_loss(self) -> float:
        """Calculates maximum position size based on daily loss limit"""
        today = datetime.now().date().isoformat()
        daily_loss = abs(min(0, self.daily_pnl.get(today, 0)))
        remaining_loss = self.current_capital * self.risk_params.max_daily_loss - daily_loss
        return remaining_loss / self.risk_params.stop_loss_distance
        
    def _get_max_position_size_by_leverage(self) -> float:
        """Calculates maximum position size based on leverage limit"""
        total_exposure = sum(p['size'] for p in self.open_positions.values())
        remaining_leverage = self.current_capital * self.risk_params.max_leverage - total_exposure
        return max(0, remaining_leverage)
        
    def _calculate_unrealized_pnl(self,
                                size: float,
                                entry_price: float,
                                current_price: float,
                                side: str) -> float:
        """Calculates unrealized P&L for a position"""
        price_change = current_price - entry_price
        if side == 'sell':
            price_change = -price_change
        return size * price_change / entry_price
        
    def _update_trailing_stop(self,
                            token_pair: str,
                            current_price: float,
                            side: str):
        """Updates trailing stop for a profitable position"""
        position = self.open_positions[token_pair]
        if position['trailing_stop'] is None:
            position['trailing_stop'] = current_price * (
                1 - self.risk_params.trailing_stop_distance
                if side == 'buy'
                else 1 + self.risk_params.trailing_stop_distance
            )
        else:
            if side == 'buy':
                position['trailing_stop'] = max(
                    position['trailing_stop'],
                    current_price * (1 - self.risk_params.trailing_stop_distance)
                )
            else:
                position['trailing_stop'] = min(
                    position['trailing_stop'],
                    current_price * (1 + self.risk_params.trailing_stop_distance)
                )
                
    def _check_stop_conditions(self,
                             token_pair: str,
                             current_price: float,
                             side: str) -> Optional[Dict]:
        """Checks if any stop conditions are triggered"""
        position = self.open_positions[token_pair]
        
        # Check hard stop loss
        if (side == 'buy' and current_price <= position['initial_stop']) or \
           (side == 'sell' and current_price >= position['initial_stop']):
            return {'reason': 'stop_loss'}
            
        # Check trailing stop
        if position['trailing_stop'] is not None:
            if (side == 'buy' and current_price <= position['trailing_stop']) or \
               (side == 'sell' and current_price >= position['trailing_stop']):
                return {'reason': 'trailing_stop'}
                
        # Check time-based stop (e.g., max holding period)
        if (datetime.now() - position['timestamp']) > timedelta(days=7):
            return {'reason': 'time_stop'}
            
        return None
        
    def _check_take_profit_conditions(self,
                                    token_pair: str,
                                    unrealized_pnl: float) -> bool:
        """Checks if take profit conditions are met"""
        position = self.open_positions[token_pair]
        profit_target = position['size'] * self.risk_params.min_profit_target
        
        return unrealized_pnl >= profit_target
        
    def _calculate_position_risk_metrics(self, token_pair: str) -> Dict:
        """Calculates comprehensive risk metrics for a position"""
        position = self.open_positions[token_pair]
        
        return {
            'position_size_ratio': position['size'] / self.current_capital,
            'time_in_trade': (datetime.now() - position['timestamp']).total_seconds(),
            'distance_to_stop': abs(
                position['trailing_stop'] - position['entry_price']
            ) / position['entry_price'] if position['trailing_stop'] else None,
            'unrealized_pnl_ratio': self._calculate_unrealized_pnl(
                position['size'],
                position['entry_price'],
                position['trailing_stop'] or position['initial_stop'],
                position['side']
            ) / self.current_capital
        }
        
    def _get_token_volatility(self, token_pair: str) -> float:
        """Gets historical volatility for a token pair"""
        # This would be implemented to fetch historical volatility
        # For now, return a default value
        return 0.2
        
    def _calculate_position_correlation(self) -> np.ndarray:
        """Calculates correlation matrix between open positions"""
        n_positions = len(self.open_positions)
        if n_positions <= 1:
            return np.array([[1.0]])
            
        # This would be implemented to calculate actual correlations
        # For now, return a simple correlation matrix
        return np.full((n_positions, n_positions), 0.5) + \
               np.eye(n_positions) * 0.5 