import asyncio
from typing import Dict, List, Optional
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import re
import json

class SentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"  # Financial domain-specific BERT
        )
        self.cache_duration = timedelta(minutes=15)
        self.sentiment_cache: Dict[str, Dict] = {}
        
    async def analyze_token_sentiment(self, token: str) -> Dict:
        """Analyzes overall sentiment for a token across multiple sources"""
        # Check cache first
        if token in self.sentiment_cache:
            cache_entry = self.sentiment_cache[token]
            if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                return cache_entry['data']
                
        # Gather sentiment from multiple sources
        twitter_sentiment = await self._analyze_twitter_sentiment(token)
        telegram_sentiment = await self._analyze_telegram_sentiment(token)
        news_sentiment = await self._analyze_news_sentiment(token)
        
        # Combine sentiment scores with weights
        combined_sentiment = self._combine_sentiment_scores({
            'twitter': (twitter_sentiment, 0.4),
            'telegram': (telegram_sentiment, 0.3),
            'news': (news_sentiment, 0.3)
        })
        
        # Cache results
        self.sentiment_cache[token] = {
            'timestamp': datetime.now(),
            'data': combined_sentiment
        }
        
        return combined_sentiment
        
    async def _analyze_twitter_sentiment(self, token: str) -> Dict:
        """Analyzes Twitter sentiment for a token"""
        async with aiohttp.ClientSession() as session:
            # This would use Twitter API in production
            # For now, return mock data
            tweets = await self._fetch_recent_tweets(session, token)
            
            if not tweets:
                return {
                    'score': 0.0,
                    'volume': 0,
                    'momentum': 0.0
                }
                
            sentiments = self.sentiment_analyzer(tweets)
            
            # Calculate aggregate metrics
            scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 
                     for s in sentiments]
            
            return {
                'score': np.mean(scores),
                'volume': len(tweets),
                'momentum': self._calculate_sentiment_momentum(scores)
            }
            
    async def _analyze_telegram_sentiment(self, token: str) -> Dict:
        """Analyzes Telegram sentiment for a token"""
        async with aiohttp.ClientSession() as session:
            # This would use Telegram API in production
            messages = await self._fetch_telegram_messages(session, token)
            
            if not messages:
                return {
                    'score': 0.0,
                    'volume': 0,
                    'momentum': 0.0
                }
                
            sentiments = self.sentiment_analyzer(messages)
            scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 
                     for s in sentiments]
            
            return {
                'score': np.mean(scores),
                'volume': len(messages),
                'momentum': self._calculate_sentiment_momentum(scores)
            }
            
    async def _analyze_news_sentiment(self, token: str) -> Dict:
        """Analyzes news sentiment for a token"""
        async with aiohttp.ClientSession() as session:
            # This would use news APIs in production
            news_articles = await self._fetch_news_articles(session, token)
            
            if not news_articles:
                return {
                    'score': 0.0,
                    'volume': 0,
                    'momentum': 0.0
                }
                
            sentiments = self.sentiment_analyzer(news_articles)
            scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 
                     for s in sentiments]
            
            return {
                'score': np.mean(scores),
                'volume': len(news_articles),
                'momentum': self._calculate_sentiment_momentum(scores)
            }
            
    def _combine_sentiment_scores(self, 
                                source_sentiments: Dict[str, tuple]) -> Dict:
        """Combines sentiment scores from different sources with weights"""
        weighted_score = 0
        total_volume = 0
        weighted_momentum = 0
        
        for source, (sentiment, weight) in source_sentiments.items():
            weighted_score += sentiment['score'] * weight
            total_volume += sentiment['volume']
            weighted_momentum += sentiment['momentum'] * weight
            
        return {
            'composite_score': weighted_score,
            'total_volume': total_volume,
            'sentiment_momentum': weighted_momentum,
            'source_breakdown': source_sentiments
        }
        
    def _calculate_sentiment_momentum(self, scores: List[float]) -> float:
        """Calculates sentiment momentum using exponential moving average"""
        if not scores:
            return 0.0
            
        # Use exponential weights for recent scores
        weights = np.exp(np.linspace(-1, 0, len(scores)))
        weights /= weights.sum()
        
        return np.average(scores, weights=weights)
        
    async def _fetch_recent_tweets(self, 
                                 session: aiohttp.ClientSession,
                                 token: str) -> List[str]:
        """Fetches recent tweets about a token"""
        # Mock implementation - would use Twitter API in production
        return [
            f"Bullish on {token}! Price looking good!",
            f"{token} showing strong momentum",
            f"Just bought more {token}, technical analysis looks promising"
        ]
        
    async def _fetch_telegram_messages(self,
                                     session: aiohttp.ClientSession,
                                     token: str) -> List[str]:
        """Fetches recent Telegram messages about a token"""
        # Mock implementation - would use Telegram API in production
        return [
            f"Great news for {token} holders!",
            f"New partnership announcement coming for {token}",
            f"Technical analysis suggests {token} is oversold"
        ]
        
    async def _fetch_news_articles(self,
                                 session: aiohttp.ClientSession,
                                 token: str) -> List[str]:
        """Fetches recent news articles about a token"""
        # Mock implementation - would use news APIs in production
        return [
            f"{token} announces major protocol upgrade",
            f"Institutional investors showing interest in {token}",
            f"Market analysis: {token} poised for growth"
        ]
        
    def get_trading_signal(self, sentiment_data: Dict) -> Dict:
        """Generates trading signal based on sentiment analysis"""
        score = sentiment_data['composite_score']
        momentum = sentiment_data['sentiment_momentum']
        volume = sentiment_data['total_volume']
        
        # Calculate signal strength
        signal_strength = (score * 0.4 + momentum * 0.4 + 
                         (np.log1p(volume) / 10) * 0.2)
        
        return {
            'action': 'buy' if signal_strength > 0 else 'sell',
            'confidence': abs(signal_strength),
            'sentiment_score': score,
            'momentum': momentum,
            'volume': volume
        } 