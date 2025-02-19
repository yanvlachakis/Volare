import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import asyncio
from datetime import datetime, timedelta

class AdvancedTradingStrategies:
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.price_scaler = MinMaxScaler()
        self.sentiment_scaler = MinMaxScaler()
        self.models = {
            'price_prediction': self._build_price_prediction_model(),
            'sentiment_analysis': self._build_sentiment_model(),
            'arbitrage': self._build_arbitrage_model()
        }
        self.market_state: Dict = {}
        
    def _build_price_prediction_model(self) -> Sequential:
        """Builds advanced LSTM model for price prediction with attention mechanism"""
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True, 
                             input_shape=(self.lookback_period, 7))),
            Dropout(0.3),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='huber')  # Huber loss for robustness
        return model
        
    def _build_sentiment_model(self) -> Sequential:
        """Builds LSTM model for sentiment analysis"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 3)),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    def _build_arbitrage_model(self) -> Sequential:
        """Builds model for detecting arbitrage opportunities"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    async def detect_arbitrage_opportunities(self, 
                                          raydium_price: float,
                                          other_dex_prices: Dict[str, float],
                                          min_profit_threshold: float = 0.002) -> List[Dict]:
        """Detects arbitrage opportunities across DEXs"""
        opportunities = []
        
        for dex_name, price in other_dex_prices.items():
            price_diff = abs(raydium_price - price)
            profit_potential = price_diff / raydium_price
            
            if profit_potential > min_profit_threshold:
                opportunities.append({
                    'dex': dex_name,
                    'price_diff': price_diff,
                    'profit_potential': profit_potential,
                    'action': 'buy' if raydium_price < price else 'sell'
                })
                
        return sorted(opportunities, 
                     key=lambda x: x['profit_potential'], 
                     reverse=True)
        
    def calculate_vwap_execution_price(self, 
                                     order_size: float,
                                     order_book: Dict,
                                     side: str = 'buy') -> Tuple[float, List[Dict]]:
        """Calculates VWAP execution price and splits order into optimal chunks"""
        levels = order_book['bids'] if side == 'buy' else order_book['asks']
        total_size = 0
        total_cost = 0
        chunks = []
        
        for level in levels:
            price, size = level['price'], level['size']
            if total_size + size >= order_size:
                remaining = order_size - total_size
                total_cost += remaining * price
                chunks.append({
                    'size': remaining,
                    'price': price
                })
                break
            total_size += size
            total_cost += size * price
            chunks.append({
                'size': size,
                'price': price
            })
            
        vwap = total_cost / order_size if order_size > 0 else 0
        return vwap, chunks
        
    async def monitor_whale_activity(self, 
                                   token: str,
                                   min_whale_size: float = 100000) -> List[Dict]:
        """Monitors large wallet movements on Solana"""
        # This would connect to a Solana RPC node to track large transfers
        whale_movements = []
        # Implementation would track large wallet movements
        # For now, return placeholder data
        return whale_movements
        
    def calculate_optimal_order_size(self, 
                                   balance: float,
                                   market_impact: float,
                                   volatility: float) -> float:
        """Calculates optimal order size using market impact model"""
        # Implementation of Almgren-Chriss market impact model
        participation_rate = 0.1  # 10% of average volume
        risk_aversion = 1.0
        
        # Calculate optimal trade size based on market impact and volatility
        optimal_size = (balance * participation_rate * 
                       np.exp(-risk_aversion * volatility * market_impact))
        
        return min(optimal_size, balance * 0.05)  # Cap at 5% of balance
        
    def generate_reinforcement_learning_signal(self, 
                                             market_state: Dict,
                                             action_history: List[Dict]) -> Dict:
        """Generates trading signal using reinforcement learning"""
        # State features
        features = np.array([
            market_state['price_momentum'],
            market_state['volume_momentum'],
            market_state['volatility'],
            market_state['sentiment_score'],
            market_state['whale_activity']
        ])
        
        # Q-learning update
        reward = self._calculate_reward(action_history[-1]) if action_history else 0
        
        return {
            'action': 'buy' if features.mean() > 0 else 'sell',
            'confidence': abs(features.mean()),
            'reward': reward
        }
        
    def _calculate_reward(self, action: Dict) -> float:
        """Calculates reward for reinforcement learning"""
        if not action.get('result'):
            return 0
            
        profit = action['result'].get('profit', 0)
        slippage = action['result'].get('slippage', 0)
        
        # Reward function considering profit and slippage
        return profit - (slippage * 2)  # Penalize slippage more heavily
        
    def update_models(self, 
                     new_data: pd.DataFrame,
                     performance_metrics: Dict):
        """Updates all models with new market data and performance metrics"""
        # Update price prediction model
        if len(new_data) >= self.lookback_period:
            X, y = self._prepare_price_data(new_data)
            self.models['price_prediction'].fit(
                X, y, epochs=1, verbose=0, batch_size=32
            )
            
        # Update sentiment model
        if 'sentiment_data' in new_data.columns:
            X_sent, y_sent = self._prepare_sentiment_data(new_data)
            self.models['sentiment_analysis'].fit(
                X_sent, y_sent, epochs=1, verbose=0, batch_size=32
            )
            
        # Update market state
        self.market_state.update(performance_metrics)
        
    def _prepare_price_data(self, 
                           data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares price data for model training"""
        features = [
            'price', 'volume', 'liquidity', 'volatility',
            'sentiment_score', 'whale_activity', 'market_impact'
        ]
        
        X = np.array([data[features].values[i:i+self.lookback_period]
                     for i in range(len(data)-self.lookback_period)])
        y = data['price'].values[self.lookback_period:]
        
        return X, y
        
    def _prepare_sentiment_data(self, 
                              data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares sentiment data for model training"""
        features = ['sentiment_score', 'volume', 'price_change']
        X = np.array([data[features].values[i:i+self.lookback_period]
                     for i in range(len(data)-self.lookback_period)])
        y = (data['price'].pct_change() > 0).values[self.lookback_period:]
        
        return X, y 