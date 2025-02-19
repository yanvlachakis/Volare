import asyncio
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
import json
import uvicorn

from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.publickey import PublicKey

from .strategies.advanced_strategies import AdvancedTradingStrategies
from .strategies.risk_management import RiskManager
from .strategies.profit_extraction import ProfitManager
from .analysis.sentiment_analyzer import SentimentAnalyzer
from .data.market_data import MarketDataPipeline
from .strategies.quantitative_strategies import QuantitativeStrategies
from .infrastructure.resource_manager import ResourceManager
from .monitoring.performance_tracker import PerformanceTracker
from .monitoring.dashboard import app, update_dashboard

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RaydiumAIBot:
    def __init__(self):
        self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        self.private_key = os.getenv("PRIVATE_KEY")
        
        if not self.private_key:
            raise ValueError("PRIVATE_KEY environment variable is required")
            
        # Initialize components
        initial_capital = float(os.getenv("START_BALANCE", "50"))  # Start with $50
        self.solana_client = AsyncClient(self.rpc_url)
        self.wallet = Keypair.from_secret_key(bytes.fromhex(self.private_key))
        self.market_data = MarketDataPipeline()
        self.advanced_strategies = AdvancedTradingStrategies()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quant_strategies = QuantitativeStrategies(initial_capital)
        
        # Initialize managers
        self.risk_manager = RiskManager(initial_capital)
        self.profit_manager = ProfitManager(initial_capital)
        self.resource_manager = ResourceManager(initial_capital)
        
        # Trading parameters
        self.min_liquidity = float(os.getenv("MIN_LIQUIDITY", "1000"))  # Lower initial requirements
        self.min_volume = float(os.getenv("MIN_VOLUME", "5000"))
        self.max_slippage = float(os.getenv("MAX_SLIPPAGE", "0.01"))
        self.min_profit_threshold = float(os.getenv("MIN_PROFIT_THRESHOLD", "0.002"))
        
        # Risk parameters
        self.max_positions = int(os.getenv("MAX_POSITIONS", "3"))  # Start with fewer positions
        self.max_positions_per_asset = int(os.getenv("MAX_POSITIONS_PER_ASSET", "1"))
        self.enable_position_netting = os.getenv("ENABLE_POSITION_NETTING", "true").lower() == "true"
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.performance_metrics: Dict = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'average_slippage': 0.0,
            'strategy_performance': {}
        }
        
        self.performance_tracker = PerformanceTracker(initial_capital=initial_capital)
        
    async def initialize(self):
        """Initialize bot and load cached data"""
        logger.info("Initializing Enhanced Raydium AI Trading Bot...")
        # Load cached market data
        for token_pair in await self.get_valid_pairs():
            cached_data = self.market_data.load_cached_data(token_pair)
            if cached_data is not None:
                self.market_data.price_cache[token_pair] = cached_data
                
    async def get_valid_pairs(self) -> List[str]:
        """Get valid trading pairs based on liquidity and volume"""
        async with self.solana_client as client:
            pairs = []
            async with client.get("https://api.raydium.io/v2/main/pairs") as response:
                data = await response.json()
                for pair in data:
                    if (float(pair["liquidity"]) > self.min_liquidity and 
                        float(pair["volume24h"]) > self.min_volume):
                        pairs.append(pair["pair"])
            return pairs
            
    async def execute_trade(self, token_pair: str, amount: float, trade_type: str) -> Dict:
        """Execute trade with advanced risk management"""
        logger.info(f"Executing {trade_type} trade for {token_pair}")
        
        # Get order book and market data
        order_book = await self._fetch_order_book(token_pair)
        market_features = self.market_data.get_market_features(token_pair)
        
        # Calculate position size with risk management
        position_size, risk_metrics = self.risk_manager.calculate_position_size(
            token_pair,
            market_features['price'],
            market_features['volatility'],
            self._estimate_market_impact(market_features)
        )
        
        # Adjust position size based on profit reinvestment
        reinvestment_amount = self.profit_manager.get_reinvestment_amount(token_pair)
        final_position_size = min(position_size, amount + reinvestment_amount)
        
        # Calculate VWAP execution price and chunks
        vwap, chunks = self.advanced_strategies.calculate_vwap_execution_price(
            final_position_size, order_book, trade_type
        )
        
        chunk_results = []
        for chunk in chunks:
            try:
                # Execute chunk trade with optimal timing
                result = await self._execute_chunk_trade(
                    token_pair,
                    chunk['size'],
                    trade_type,
                    chunk['price']
                )
                chunk_results.append(result)
                
                # Dynamic sleep between chunks based on market activity
                sleep_time = self._calculate_optimal_sleep_time(market_features)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error executing chunk trade: {str(e)}")
                continue
                
        # Aggregate results and update tracking
        trade_result = self._aggregate_trade_results(chunk_results)
        self._update_performance_metrics(trade_result)
        
        # Process profit and update risk management
        profit_allocation = self.profit_manager.process_trade_profit(trade_result)
        self.risk_manager.update_position(token_pair, trade_result)
        
        # Record trade result
        trade_result['risk_metrics'] = risk_metrics
        trade_result['profit_allocation'] = profit_allocation
        
        await self.performance_tracker.update_trade_metrics(trade_result)
        
        # Check if we need to adjust strategy allocation based on performance
        if self.performance_tracker.get_strategy_metrics(trade_result['strategy'])['win_rate'] < 0.5:
            await self._adjust_strategy_allocation(trade_result['strategy'])
        
        return trade_result
        
    async def _execute_chunk_trade(self,
                                 token_pair: str,
                                 amount: float,
                                 trade_type: str,
                                 target_price: float) -> Dict:
        """Execute a single chunk of a trade with price optimization"""
        # Implement actual Raydium swap logic here
        # This is a placeholder for the actual implementation
        return {
            "success": True,
            "amount": amount,
            "price": target_price,
            "timestamp": datetime.now(),
            "slippage": 0.001  # Placeholder
        }
        
    def _aggregate_trade_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple chunk trades with advanced metrics"""
        if not results:
            return {"success": False, "error": "No successful trades"}
            
        total_amount = sum(r["amount"] for r in results)
        weighted_price = sum(r["amount"] * r["price"] for r in results) / total_amount
        avg_slippage = sum(r["slippage"] for r in results) / len(results)
        
        return {
            "success": True,
            "total_amount": total_amount,
            "average_price": weighted_price,
            "slippage": avg_slippage,
            "chunks_completed": len(results),
            "execution_time": (results[-1]["timestamp"] - results[0]["timestamp"]).total_seconds()
        }
        
    async def run_trading_loop(self):
        """Enhanced trading loop with dynamic strategy allocation"""
        logger.info("Starting enhanced AI-driven trading loop...")
        
        while True:
            try:
                # Optimize resources based on current portfolio
                portfolio_value = await self._get_portfolio_value()
                await self.resource_manager.optimize_resources(portfolio_value)
                
                # Get optimal RPC endpoint
                optimal_rpc = await self.resource_manager.get_optimal_rpc_endpoint()
                self.solana_client = AsyncClient(optimal_rpc)
                
                # Update portfolio metrics
                current_drawdown = self._calculate_drawdown(portfolio_value)
                self.profit_manager.manage_drawdown(current_drawdown)
                
                if self._should_reduce_risk(current_drawdown):
                    await self._reduce_risk_exposure()
                    
                # Process valid trading pairs
                valid_pairs = await self.get_valid_pairs()
                for token_pair in valid_pairs:
                    if len(self.risk_manager.open_positions) >= self.max_positions:
                        logger.info("Maximum positions reached, skipping new trades")
                        break
                        
                    # Update market data
                    await self.market_data.update_price_cache(token_pair)
                    if self.market_data.detect_anomalies(token_pair):
                        continue
                        
                    # Get market features and order book
                    features = self.market_data.get_market_features(token_pair)
                    order_book = await self._fetch_order_book(token_pair)
                    if not features or not order_book:
                        continue
                        
                    # Execute HFT strategy if conditions are right
                    hft_signal = await self.quant_strategies.execute_hft_strategy(
                        token_pair, order_book, features
                    )
                    if hft_signal:
                        await self.execute_trade(
                            token_pair,
                            hft_signal['size'],
                            hft_signal['action']
                        )
                        continue
                        
                    # Try market making
                    mm_orders = await self.quant_strategies.execute_market_making(
                        token_pair, order_book, features
                    )
                    if mm_orders:
                        for order in mm_orders:
                            await self.execute_trade(
                                token_pair,
                                order['size'],
                                order['action']
                            )
                        continue
                        
                    # Check for statistical arbitrage opportunities
                    stat_arb = await self.quant_strategies.execute_stat_arb(
                        valid_pairs,
                        self.market_data.price_cache
                    )
                    if stat_arb:
                        # Execute stat arb trades
                        await self.execute_trade(
                            stat_arb['pair_a'],
                            stat_arb['size_a'],
                            stat_arb['action_a']
                        )
                        continue
                        
                    # If no quant opportunities, try AI-driven strategies
                    sentiment = await self.sentiment_analyzer.analyze_token_sentiment(
                        token_pair.split('/')[0]
                    )
                    sentiment_signal = self.sentiment_analyzer.get_trading_signal(sentiment)
                    
                    price_prediction, confidence = self.advanced_strategies.predict_price_movement(
                        self.market_data.price_cache[token_pair]
                    )
                    
                    whale_activity = await self.advanced_strategies.monitor_whale_activity(
                        token_pair.split('/')[0]
                    )
                    
                    # Make trade decision
                    trade_decision = self._make_trade_decision(
                        price_prediction,
                        confidence,
                        sentiment_signal,
                        whale_activity,
                        features
                    )
                    
                    if trade_decision['should_trade']:
                        position_size = self.risk_manager.calculate_position_size(
                            token_pair,
                            features['price'],
                            features['volatility'],
                            trade_decision['market_impact']
                        )[0]
                        
                        await self.execute_trade(
                            token_pair,
                            position_size,
                            trade_decision['action']
                        )
                        
                # Manage open positions
                await self._manage_open_positions()
                
                # Extract profits if conditions met
                await self._process_profit_extraction(portfolio_value)
                
                # Update performance metrics
                self._update_strategy_performance()
                
                # Sleep between iterations (dynamic based on strategy)
                sleep_time = self._calculate_optimal_sleep_time(features)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(60)
                
    async def _manage_open_positions(self):
        """Manages all open positions"""
        for token_pair in list(self.risk_manager.open_positions.keys()):
            try:
                # Get current market data
                features = self.market_data.get_market_features(token_pair)
                if not features:
                    continue
                    
                # Check position status
                position_action = self.risk_manager.manage_open_position(
                    token_pair,
                    features['price'],
                    self.risk_manager.open_positions[token_pair]
                )
                
                # Execute position management action
                if position_action['action'] in ['close', 'reduce']:
                    await self.execute_trade(
                        token_pair,
                        position_action.get('size_reduction', 
                                          self.risk_manager.open_positions[token_pair]['size']),
                        'sell' if self.risk_manager.open_positions[token_pair]['side'] == 'buy' 
                        else 'buy'
                    )
                    
            except Exception as e:
                logger.error(f"Error managing position for {token_pair}: {str(e)}")
                
    async def _process_profit_extraction(self, portfolio_value: float):
        """Processes profit extraction if conditions are met"""
        extraction_result = self.profit_manager.extract_profits(portfolio_value)
        
        if extraction_result['extracted'] > 0:
            logger.info(
                f"Extracted {extraction_result['extracted']:.2f} in profits, "
                f"reinvested {extraction_result['reinvested']:.2f}"
            )
            
    def _check_risk_limits(self, token_pair: str, trade_decision: Dict) -> bool:
        """Checks if trade meets risk management criteria"""
        # Check position limits
        token_positions = sum(
            1 for pos in self.risk_manager.open_positions.items()
            if pos[0].split('/')[0] == token_pair.split('/')[0]
        )
        
        if token_positions >= self.max_positions_per_asset:
            return False
            
        # Check portfolio risk
        portfolio_risk = self.risk_manager._calculate_portfolio_risk()
        if portfolio_risk >= self.risk_manager.risk_params.max_drawdown:
            return False
            
        return True
        
    def _should_reduce_risk(self, current_drawdown: float) -> bool:
        """Determines if risk exposure should be reduced"""
        return (current_drawdown > self.risk_manager.risk_params.max_drawdown * 0.8 or
                len(self.risk_manager.open_positions) > self.max_positions * 0.8)
        
    async def _reduce_risk_exposure(self):
        """Reduces risk exposure by closing or reducing positions"""
        # Sort positions by risk level
        positions = sorted(
            self.risk_manager.open_positions.items(),
            key=lambda x: x[1]['size'] * self._get_token_volatility(x[0]),
            reverse=True
        )
        
        # Close or reduce highest risk positions
        for token_pair, position in positions[:2]:  # Reduce top 2 risky positions
            await self.execute_trade(
                token_pair,
                position['size'] * 0.5,  # Reduce by 50%
                'sell' if position['side'] == 'buy' else 'buy'
            )
            
    async def _get_portfolio_value(self) -> float:
        """Calculates total portfolio value"""
        # This would be implemented to get actual portfolio value
        # For now, return a simple calculation
        return self.risk_manager.current_capital
        
    def _calculate_drawdown(self, current_value: float) -> float:
        """Calculates current drawdown from peak"""
        return (self.risk_manager.peak_capital - current_value) / self.risk_manager.peak_capital
        
    def _get_token_volatility(self, token_pair: str) -> float:
        """Gets token volatility from market data"""
        features = self.market_data.get_market_features(token_pair)
        return features.get('volatility', 0.2) if features else 0.2
        
    async def _fetch_order_book(self, token_pair: str) -> Dict:
        """Fetches order book data from Raydium"""
        # Implement actual order book fetching logic
        # This is a placeholder
        return {
            'bids': [{'price': 100, 'size': 1000}],
            'asks': [{'price': 101, 'size': 1000}]
        }
        
    async def _fetch_other_dex_prices(self, token_pair: str) -> Dict[str, float]:
        """Fetches prices from other DEXs for arbitrage"""
        # Implement actual price fetching from other DEXs
        # This is a placeholder
        return {
            'serum': 100.0,
            'orca': 100.1
        }
        
    async def _execute_arbitrage(self, opportunity: Dict):
        """Executes an arbitrage opportunity"""
        logger.info(f"Executing arbitrage opportunity: {opportunity}")
        # Implement actual arbitrage execution logic
        pass
        
    def _make_trade_decision(self,
                           price_prediction: float,
                           confidence: float,
                           sentiment_signal: Dict,
                           whale_activity: List[Dict],
                           market_features: Dict) -> Dict:
        """Makes final trade decision combining all signals"""
        # Combine multiple signals with weights
        price_signal = 1 if price_prediction > market_features['price'] else -1
        sentiment_weight = 0.3
        price_weight = 0.4
        whale_weight = 0.3
        
        # Calculate whale signal
        whale_signal = sum(1 if w['action'] == 'buy' else -1 for w in whale_activity)
        whale_signal = np.tanh(whale_signal)  # Normalize to [-1, 1]
        
        # Combined signal
        weighted_signal = (
            price_signal * price_weight * confidence +
            sentiment_signal['confidence'] * sentiment_weight * 
            (1 if sentiment_signal['action'] == 'buy' else -1) +
            whale_signal * whale_weight
        )
        
        return {
            'should_trade': abs(weighted_signal) > 0.5,
            'action': 'buy' if weighted_signal > 0 else 'sell',
            'confidence': abs(weighted_signal),
            'market_impact': self._estimate_market_impact(market_features)
        }
        
    def _estimate_market_impact(self, market_features: Dict) -> float:
        """Estimates market impact using square root law"""
        # Implementation of square root market impact model
        volatility = market_features['volatility']
        volume = market_features['volume']
        
        return 0.1 * volatility * np.sqrt(1.0 / volume)
        
    def _calculate_optimal_sleep_time(self, features: Dict) -> float:
        """Calculate optimal sleep time between iterations"""
        base_sleep = 1.0  # 1 second base
        
        # Adjust based on volatility
        volatility = features.get('volatility', 0.1)
        volume = features.get('volume', 0)
        
        # More frequent updates in high volatility/volume
        sleep_time = base_sleep * (1 - min(volatility, 0.5))
        
        # Ensure minimum sleep time
        return max(0.1, min(sleep_time, 5.0))  # Between 0.1 and 5 seconds
        
    def _update_performance_metrics(self, trade_result: Dict):
        """Updates performance tracking metrics"""
        self.performance_metrics['total_trades'] += 1
        if trade_result['success']:
            self.performance_metrics['successful_trades'] += 1
            self.performance_metrics['total_profit'] += trade_result.get('profit', 0)
            self.performance_metrics['average_slippage'] = (
                (self.performance_metrics['average_slippage'] * 
                 (self.performance_metrics['total_trades'] - 1) +
                 trade_result['slippage']) / 
                self.performance_metrics['total_trades']
            )
            
    def _update_strategy_performance(self):
        """Update performance metrics for each strategy"""
        for strategy_type in ['hft', 'market_making', 'stat_arb', 'ai_driven']:
            strategy_trades = [
                trade for trade in self.trade_history
                if trade.get('strategy_type') == strategy_type
            ]
            
            if strategy_trades:
                self.performance_metrics['strategy_performance'][strategy_type] = {
                    'total_trades': len(strategy_trades),
                    'successful_trades': sum(1 for t in strategy_trades if t.get('profit', 0) > 0),
                    'total_profit': sum(t.get('profit', 0) for t in strategy_trades),
                    'average_return': (sum(t.get('profit', 0) for t in strategy_trades) / 
                                     len(strategy_trades))
                }
                
    async def _monitor_resources(self):
        """Monitor system resources and costs"""
        while True:
            try:
                metrics = {
                    'cpu': self.resource_manager.get_cpu_usage(),
                    'memory': self.resource_manager.get_memory_usage(),
                    'network': self.resource_manager.get_network_usage(),
                    'cost': self.resource_manager.get_current_costs()
                }
                
                await self.performance_tracker.update_resource_metrics(metrics)
                
                # Adjust resources based on performance and usage
                if self.performance_tracker.get_performance_summary()['portfolio']['current_capital'] > 100:
                    await self.resource_manager.scale_resources(scale_up=True)
                    
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def start(self):
        """Start the trading bot with monitoring"""
        try:
            # Start resource monitoring in background
            asyncio.create_task(self._monitor_resources())
            
            while True:
                # Get performance metrics
                performance = self.performance_tracker.get_performance_summary()
                logger.info(f"Current performance: {json.dumps(performance, indent=2)}")
                
                # Execute trading strategies
                for strategy in self.active_strategies:
                    strategy_metrics = self.performance_tracker.get_strategy_metrics(strategy)
                    
                    # Skip underperforming strategies
                    if strategy_metrics.get('win_rate', 0) < 0.4:
                        continue
                        
                    await self.execute_strategy(strategy)
                    
                await asyncio.sleep(1)  # Trading loop interval
                
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
            raise

    async def _adjust_strategy_allocation(self, strategy: str):
        """Adjust strategy allocation based on performance"""
        metrics = self.performance_tracker.get_strategy_metrics(strategy)
        
        if metrics['win_rate'] < 0.4:
            # Reduce allocation for underperforming strategy
            self.strategy_allocations[strategy] *= 0.5
        elif metrics['win_rate'] > 0.6:
            # Increase allocation for well-performing strategy
            self.strategy_allocations[strategy] = min(
                self.strategy_allocations[strategy] * 1.5,
                0.5  # Maximum 50% allocation per strategy
            )

async def main():
    """Main entry point for the trading bot"""
    try:
        # Initialize bot
        config = load_config()
        bot = RaydiumAIBot(config)
        await bot.initialize()
        
        # Start dashboard update task
        dashboard_task = asyncio.create_task(
            update_dashboard(bot.performance_tracker)
        )
        
        # Start FastAPI server in the background
        server_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(server_config)
        server_task = asyncio.create_task(server.serve())
        
        # Start bot
        await bot.start()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'dashboard_task' in locals():
            dashboard_task.cancel()
        if 'server_task' in locals():
            server_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 