import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import os
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    min_profit_threshold: float = 0.002  # 0.2% minimum profit
    max_drawdown_threshold: float = 0.1  # 10% maximum drawdown
    min_win_rate: float = 0.55  # 55% minimum win rate
    max_daily_loss: float = 0.05  # 5% maximum daily loss
    alert_cooldown: int = 300  # 5 minutes between alerts

class PerformanceTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.metrics = PerformanceMetrics()
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.strategy_performance: Dict[str, Dict] = {}
        self.resource_usage: Dict[str, List[float]] = {
            'cpu': [],
            'memory': [],
            'network': [],
            'cost': []
        }
        
        # Alert tracking
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_history: List[Dict] = []
        
    async def update_trade_metrics(self, trade_result: Dict):
        """Update performance metrics with new trade result"""
        self.trade_history.append({
            **trade_result,
            'timestamp': datetime.now()
        })
        
        # Update capital
        self.current_capital += trade_result.get('pnl', 0)
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Update daily P&L
        today = datetime.now().date().isoformat()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + trade_result.get('pnl', 0)
        
        # Update strategy performance
        strategy = trade_result.get('strategy', 'unknown')
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'max_drawdown': 0
            }
            
        perf = self.strategy_performance[strategy]
        perf['total_trades'] += 1
        perf['winning_trades'] += 1 if trade_result.get('pnl', 0) > 0 else 0
        perf['total_pnl'] += trade_result.get('pnl', 0)
        
        # Check for alerts
        await self._check_performance_alerts()
        
    async def update_resource_metrics(self, metrics: Dict):
        """Update resource utilization metrics"""
        for key in self.resource_usage:
            if key in metrics:
                self.resource_usage[key].append(metrics[key])
                # Keep last 24 hours of data
                if len(self.resource_usage[key]) > 1440:  # 24h * 60min
                    self.resource_usage[key].pop(0)
                    
        # Check resource alerts
        await self._check_resource_alerts()
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'portfolio': {
                'current_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / 
                               self.initial_capital,
                'peak_capital': self.peak_capital,
                'max_drawdown': self._calculate_max_drawdown()
            },
            'trading': {
                'total_trades': len(self.trade_history),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'average_trade': self._calculate_average_trade()
            },
            'strategies': self.strategy_performance,
            'resource_usage': {
                key: np.mean(values[-60:])  # Last hour average
                for key, values in self.resource_usage.items()
            }
        }
        
    def get_strategy_metrics(self, strategy: str) -> Dict:
        """Get detailed metrics for a specific strategy"""
        if strategy not in self.strategy_performance:
            return {}
            
        perf = self.strategy_performance[strategy]
        trades = [t for t in self.trade_history if t.get('strategy') == strategy]
        
        return {
            **perf,
            'win_rate': perf['winning_trades'] / perf['total_trades'],
            'average_trade': perf['total_pnl'] / perf['total_trades'],
            'sharpe_ratio': self._calculate_sharpe_ratio(trades),
            'max_drawdown': self._calculate_strategy_drawdown(trades)
        }
        
    async def _check_performance_alerts(self):
        """Check and generate performance-related alerts"""
        # Check drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.metrics.max_drawdown_threshold:
            await self._generate_alert(
                'drawdown',
                f"Drawdown exceeded threshold: {current_drawdown:.2%}"
            )
            
        # Check daily loss
        today = datetime.now().date().isoformat()
        daily_loss = abs(min(0, self.daily_pnl.get(today, 0)))
        if daily_loss > self.initial_capital * self.metrics.max_daily_loss:
            await self._generate_alert(
                'daily_loss',
                f"Daily loss exceeded threshold: ${daily_loss:.2f}"
            )
            
        # Check win rate
        win_rate = self._calculate_win_rate()
        if win_rate < self.metrics.min_win_rate:
            await self._generate_alert(
                'win_rate',
                f"Win rate below threshold: {win_rate:.2%}"
            )
            
    async def _check_resource_alerts(self):
        """Check and generate resource-related alerts"""
        if self.resource_usage['cpu']:
            cpu_usage = np.mean(self.resource_usage['cpu'][-5:])  # Last 5 minutes
            if cpu_usage > 0.8:  # 80% threshold
                await self._generate_alert(
                    'cpu_usage',
                    f"High CPU usage: {cpu_usage:.2%}"
                )
                
        if self.resource_usage['memory']:
            memory_usage = np.mean(self.resource_usage['memory'][-5:])
            if memory_usage > 0.8:
                await self._generate_alert(
                    'memory_usage',
                    f"High memory usage: {memory_usage:.2%}"
                )
                
        if self.resource_usage['cost']:
            monthly_cost = sum(self.resource_usage['cost'])
            if monthly_cost > 45:  # Alert at 90% of budget
                await self._generate_alert(
                    'cost',
                    f"Monthly cost approaching budget: ${monthly_cost:.2f}"
                )
                
    async def _generate_alert(self, alert_type: str, message: str):
        """Generate and log alerts with cooldown"""
        now = datetime.now()
        if alert_type in self.last_alert_time:
            if (now - self.last_alert_time[alert_type]).seconds < self.metrics.alert_cooldown:
                return
                
        self.last_alert_time[alert_type] = now
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': now,
            'portfolio_value': self.current_capital
        }
        
        self.alert_history.append(alert)
        logger.warning(f"Alert: {message}")
        
        # Implement notification logic here (e.g., Telegram, Discord)
        
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        if not self.trade_history:
            return 0.0
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trade_history)
        
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
    def _calculate_average_trade(self) -> float:
        """Calculate average trade profit/loss"""
        if not self.trade_history:
            return 0.0
        return sum(t.get('pnl', 0) for t in self.trade_history) / len(self.trade_history)
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trade_history:
            return 0.0
            
        peak = self.initial_capital
        max_drawdown = 0.0
        current = self.initial_capital
        
        for trade in self.trade_history:
            current += trade.get('pnl', 0)
            peak = max(peak, current)
            drawdown = (peak - current) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
        
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio for a set of trades"""
        if not trades:
            return 0.0
            
        returns = [t.get('pnl', 0) / self.initial_capital for t in trades]
        if not returns:
            return 0.0
            
        return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365)  # Annualized
        
    def _calculate_strategy_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown for a specific strategy"""
        if not trades:
            return 0.0
            
        peak = 0
        current = 0
        max_drawdown = 0.0
        
        for trade in trades:
            current += trade.get('pnl', 0)
            peak = max(peak, current)
            drawdown = (peak - current) / (peak + 1e-10)
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
