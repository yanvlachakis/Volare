import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProfitParameters:
    min_profit_threshold: float
    reinvestment_ratio: float
    profit_taking_intervals: List[float]
    max_reinvestment_amount: float
    reserve_ratio: float
    compound_frequency: int
    max_drawdown_reset: float
    profit_lock_period: int

class ProfitManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.locked_profits = 0.0
        self.available_profits = 0.0
        self.reinvestment_pool = 0.0
        self.profit_history: List[Dict] = []
        self.last_compound_date = datetime.now()
        
        # Load profit parameters
        self.profit_params = ProfitParameters(
            min_profit_threshold=0.01,    # 1% minimum profit to extract
            reinvestment_ratio=0.6,       # 60% of profits for reinvestment
            profit_taking_intervals=[0.02, 0.05, 0.10],  # Take profits at 2%, 5%, 10%
            max_reinvestment_amount=50000,  # Maximum amount for reinvestment
            reserve_ratio=0.2,             # 20% of profits to reserve
            compound_frequency=7,          # Compound every 7 days
            max_drawdown_reset=0.1,       # Reset profit lock at 10% drawdown
            profit_lock_period=30         # Lock profits for 30 days
        )
        
    def process_trade_profit(self, trade_result: Dict) -> Dict:
        """Processes and allocates profit from a trade"""
        if not trade_result.get('success', False):
            return self._create_profit_allocation(0.0)
            
        profit = trade_result.get('pnl', 0.0)
        if profit <= 0:
            return self._create_profit_allocation(0.0)
            
        # Calculate profit allocation
        profit_allocation = self._allocate_profit(profit)
        
        # Update profit tracking
        self._update_profit_tracking(profit, profit_allocation)
        
        # Check if it's time to compound profits
        if self._should_compound_profits():
            self._compound_profits()
            
        return profit_allocation
        
    def get_reinvestment_amount(self, token_pair: str) -> float:
        """Calculates available reinvestment amount for a new trade"""
        if self.reinvestment_pool <= 0:
            return 0.0
            
        # Consider token-specific factors
        token_metrics = self._get_token_performance_metrics(token_pair)
        
        # Calculate base reinvestment amount
        base_amount = min(
            self.reinvestment_pool,
            self.profit_params.max_reinvestment_amount
        )
        
        # Adjust based on token performance
        adjusted_amount = base_amount * token_metrics['performance_factor']
        
        return min(adjusted_amount, self.reinvestment_pool)
        
    def extract_profits(self, current_portfolio_value: float) -> Dict:
        """Extracts profits based on predefined rules"""
        total_profit = current_portfolio_value - self.initial_capital
        if total_profit <= 0:
            return {'extracted': 0.0, 'reinvested': 0.0}
            
        # Check if we've hit any profit-taking intervals
        for interval in sorted(self.profit_params.profit_taking_intervals):
            if total_profit >= (self.initial_capital * interval):
                extraction_amount = self._calculate_extraction_amount(
                    total_profit, interval
                )
                
                # Update profit tracking
                self.locked_profits += extraction_amount * (1 - self.profit_params.reinvestment_ratio)
                self.reinvestment_pool += extraction_amount * self.profit_params.reinvestment_ratio
                
                return {
                    'extracted': extraction_amount * (1 - self.profit_params.reinvestment_ratio),
                    'reinvested': extraction_amount * self.profit_params.reinvestment_ratio
                }
                
        return {'extracted': 0.0, 'reinvested': 0.0}
        
    def manage_drawdown(self, current_drawdown: float):
        """Manages profit allocation during drawdown periods"""
        if current_drawdown >= self.profit_params.max_drawdown_reset:
            # Reset profit locks during significant drawdown
            self.locked_profits = 0.0
            self.reinvestment_pool = 0.0
            logger.warning(f"Profit locks reset due to {current_drawdown:.2%} drawdown")
            
    def _allocate_profit(self, profit: float) -> Dict:
        """Allocates profit between different pools"""
        if profit < self.initial_capital * self.profit_params.min_profit_threshold:
            return self._create_profit_allocation(profit)
            
        # Calculate allocations
        reinvestment = profit * self.profit_params.reinvestment_ratio
        reserve = profit * self.profit_params.reserve_ratio
        available = profit * (1 - self.profit_params.reinvestment_ratio - 
                            self.profit_params.reserve_ratio)
        
        return {
            'total_profit': profit,
            'reinvestment': reinvestment,
            'reserve': reserve,
            'available': available,
            'timestamp': datetime.now()
        }
        
    def _update_profit_tracking(self, profit: float, allocation: Dict):
        """Updates profit tracking with new profit data"""
        self.profit_history.append({
            'timestamp': datetime.now(),
            'profit': profit,
            'allocation': allocation,
            'portfolio_value': self.current_capital
        })
        
        # Update profit pools
        self.reinvestment_pool += allocation['reinvestment']
        self.available_profits += allocation['available']
        self.locked_profits += allocation['reserve']
        
    def _should_compound_profits(self) -> bool:
        """Determines if it's time to compound profits"""
        days_since_last_compound = (
            datetime.now() - self.last_compound_date
        ).days
        
        return days_since_last_compound >= self.profit_params.compound_frequency
        
    def _compound_profits(self):
        """Compounds available profits back into trading capital"""
        if self.reinvestment_pool <= 0:
            return
            
        # Calculate compound amount
        compound_amount = min(
            self.reinvestment_pool,
            self.profit_params.max_reinvestment_amount
        )
        
        # Update capital and pools
        self.current_capital += compound_amount
        self.reinvestment_pool -= compound_amount
        self.last_compound_date = datetime.now()
        
        logger.info(f"Compounded {compound_amount:.2f} into trading capital")
        
    def _calculate_extraction_amount(self, total_profit: float, interval: float) -> float:
        """Calculates how much profit to extract at a given interval"""
        interval_profit = self.initial_capital * interval
        extraction_ratio = min(1.0, total_profit / interval_profit)
        
        return total_profit * extraction_ratio
        
    def _get_token_performance_metrics(self, token_pair: str) -> Dict:
        """Calculates performance metrics for a specific token"""
        # This would be implemented to calculate actual metrics
        # For now, return default values
        return {
            'performance_factor': 0.8,  # Conservative factor
            'risk_score': 0.5,
            'historical_profit_ratio': 1.2
        }
        
    def _create_profit_allocation(self, profit: float) -> Dict:
        """Creates a basic profit allocation structure"""
        return {
            'total_profit': profit,
            'reinvestment': 0.0,
            'reserve': 0.0,
            'available': profit,
            'timestamp': datetime.now()
        }
        
    def get_profit_metrics(self) -> Dict:
        """Returns current profit metrics"""
        return {
            'total_locked_profits': self.locked_profits,
            'available_profits': self.available_profits,
            'reinvestment_pool': self.reinvestment_pool,
            'total_profit': (self.current_capital - self.initial_capital +
                           self.locked_profits),
            'roi': (self.current_capital - self.initial_capital) / 
                   self.initial_capital,
            'compound_status': {
                'last_compound': self.last_compound_date,
                'next_compound': self.last_compound_date + 
                                timedelta(days=self.profit_params.compound_frequency)
            }
        } 