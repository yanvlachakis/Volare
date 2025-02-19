import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple

class AITradingStrategy:
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.price_scaler = MinMaxScaler()
        self.model = self._build_lstm_model()
        self.market_state: Dict = {}
        
    def _build_lstm_model(self) -> Sequential:
        """Builds LSTM model for price prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_data(self, price_data: pd.DataFrame) -> np.ndarray:
        """Prepares data for LSTM model"""
        features = ['price', 'volume', 'liquidity', 'volatility', 'market_cap']
        scaled_data = self.price_scaler.fit_transform(price_data[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_period):
            X.append(scaled_data[i:(i + self.lookback_period)])
            y.append(scaled_data[i + self.lookback_period, 0])
        return np.array(X), np.array(y)
    
    def predict_price_movement(self, recent_data: pd.DataFrame) -> Tuple[float, float]:
        """Predicts next price movement and confidence score"""
        X, _ = self.prepare_data(recent_data)
        if len(X) > 0:
            prediction = self.model.predict(X[-1:])
            confidence = self._calculate_prediction_confidence(prediction[0][0])
            return float(prediction[0][0]), confidence
        return 0.0, 0.0
    
    def _calculate_prediction_confidence(self, prediction: float) -> float:
        """Calculates confidence score based on prediction stability"""
        # Implement confidence calculation based on prediction variance
        return min(abs(prediction - 0.5) * 2, 1.0)
    
    def select_strategy(self, token_data: Dict) -> str:
        """Selects optimal trading strategy based on token metrics"""
        mcap = token_data.get('market_cap', 0)
        volume = token_data.get('volume24h', 0)
        volatility = token_data.get('volatility', 0)
        
        if mcap < 1_000_000 and volatility > 0.1:
            return "scalp_trading"
        elif mcap < 10_000_000 and volume > 100_000:
            return "momentum_trading"
        else:
            return "statistical_arbitrage"
    
    def calculate_position_size(self, balance: float, token_data: Dict) -> float:
        """Calculates optimal position size using Kelly Criterion"""
        win_rate = self.market_state.get('win_rate', 0.5)
        profit_ratio = self.market_state.get('avg_profit_ratio', 1.5)
        loss_ratio = self.market_state.get('avg_loss_ratio', 1.0)
        
        kelly_fraction = (win_rate * profit_ratio - (1 - win_rate)) / profit_ratio
        # Limit position size to 5% of balance for risk management
        return min(balance * max(kelly_fraction, 0), balance * 0.05)
    
    def update_market_state(self, trade_result: Dict):
        """Updates market state with latest trade results"""
        self.market_state.update({
            'last_trade_profit': trade_result.get('profit', 0),
            'win_rate': trade_result.get('win_rate', self.market_state.get('win_rate', 0.5)),
            'avg_profit_ratio': trade_result.get('profit_ratio', self.market_state.get('avg_profit_ratio', 1.5)),
            'avg_loss_ratio': trade_result.get('loss_ratio', self.market_state.get('avg_loss_ratio', 1.0))
        }) 