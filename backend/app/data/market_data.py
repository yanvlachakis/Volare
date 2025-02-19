import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import aiohttp
import asyncio
from datetime import datetime, timedelta
import joblib
from pathlib import Path

class MarketDataPipeline:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.volatility_window = 24  # hours
        
    async def fetch_market_data(self, token_pair: str) -> Dict:
        """Fetches real-time market data from Raydium"""
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.raydium.io/v2/main/pairs") as response:
                data = await response.json()
                for pair in data:
                    if pair["pair"] == token_pair:
                        return {
                            "price": float(pair["price"]),
                            "volume24h": float(pair["volume24h"]),
                            "liquidity": float(pair["liquidity"]),
                            "price_change24h": float(pair.get("price_change_24h", 0))
                        }
        return {}
    
    def calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculates rolling volatility using log returns"""
        if len(price_data) < 2:
            return 0.0
        log_returns = np.log(price_data['price'] / price_data['price'].shift(1))
        return log_returns.std() * np.sqrt(self.volatility_window)
    
    async def update_price_cache(self, token_pair: str):
        """Updates price cache with latest market data"""
        market_data = await self.fetch_market_data(token_pair)
        if not market_data:
            return
        
        timestamp = datetime.now()
        new_data = pd.DataFrame([{
            'timestamp': timestamp,
            'price': market_data['price'],
            'volume': market_data['volume24h'],
            'liquidity': market_data['liquidity']
        }])
        
        if token_pair in self.price_cache:
            self.price_cache[token_pair] = pd.concat([
                self.price_cache[token_pair],
                new_data
            ]).last('24H')
        else:
            self.price_cache[token_pair] = new_data
            
        # Save to disk cache
        cache_file = self.cache_dir / f"{token_pair.replace('/', '_')}_cache.pkl"
        joblib.dump(self.price_cache[token_pair], cache_file)
    
    def get_market_features(self, token_pair: str) -> Dict:
        """Extracts market features for ML model"""
        if token_pair not in self.price_cache:
            return {}
            
        df = self.price_cache[token_pair]
        if len(df) < 2:
            return {}
            
        latest_data = df.iloc[-1]
        volatility = self.calculate_volatility(df)
        
        return {
            'price': latest_data['price'],
            'volume': latest_data['volume'],
            'liquidity': latest_data['liquidity'],
            'volatility': volatility,
            'price_momentum': df['price'].pct_change().mean(),
            'volume_momentum': df['volume'].pct_change().mean()
        }
    
    def detect_anomalies(self, token_pair: str) -> bool:
        """Detects potential pump & dump patterns"""
        if token_pair not in self.price_cache:
            return False
            
        df = self.price_cache[token_pair]
        if len(df) < 24:  # Need at least 24 data points
            return False
            
        price_changes = df['price'].pct_change()
        volume_changes = df['volume'].pct_change()
        
        # Suspicious patterns:
        # 1. Sudden price spike with volume spike
        # 2. Sharp price increase followed by decrease
        price_volatility = price_changes.std()
        volume_volatility = volume_changes.std()
        
        is_suspicious = (
            (abs(price_changes.iloc[-1]) > 3 * price_volatility and 
             abs(volume_changes.iloc[-1]) > 3 * volume_volatility) or
            (price_changes.iloc[-2:].mean() > 0.1 and 
             price_changes.iloc[-1] < -0.05)
        )
        
        return is_suspicious
    
    def load_cached_data(self, token_pair: str) -> Optional[pd.DataFrame]:
        """Loads cached market data from disk"""
        cache_file = self.cache_dir / f"{token_pair.replace('/', '_')}_cache.pkl"
        if cache_file.exists():
            return joblib.load(cache_file)
        return None 