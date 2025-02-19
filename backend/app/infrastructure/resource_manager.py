import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class ResourceConfig:
    max_monthly_budget: float = 50.0  # Maximum monthly infrastructure cost
    min_rpc_nodes: int = 1
    max_rpc_nodes: int = 3
    rpc_cost_per_node: float = 15.0  # Cost per RPC node per month
    min_portfolio_size: float = 50.0  # Minimum portfolio size to operate
    scaling_threshold: float = 0.2  # 20% portfolio growth triggers scaling
    cpu_threshold: float = 0.8  # CPU utilization threshold for scaling
    memory_threshold: float = 0.8  # Memory utilization threshold
    backup_rpc_urls: list = None  # Default to None, will be set in __post_init__

    def __post_init__(self):
        self.backup_rpc_urls = [
            "https://api.mainnet-beta.solana.com",
            "https://solana-api.projectserum.com",
            "https://rpc.ankr.com/solana"
        ]

class ResourceManager:
    def __init__(self, initial_portfolio_size: float):
        self.portfolio_size = initial_portfolio_size
        self.config = ResourceConfig()
        self.active_rpc_nodes: Dict[str, Dict] = {}
        self.resource_metrics: Dict[str, float] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'network_latency': 0.0,
            'monthly_cost': 0.0
        }
        self.last_scaling_check = datetime.now()
        self.performance_history: list = []
        
    async def optimize_resources(self, current_portfolio_size: float):
        """Optimize resource allocation based on portfolio size and performance"""
        self.portfolio_size = current_portfolio_size
        
        # Check if we need to scale
        if self._should_check_scaling():
            await self._evaluate_scaling_needs()
            
        # Optimize RPC node usage
        await self._optimize_rpc_nodes()
        
        # Update resource metrics
        await self._update_resource_metrics()
        
    async def get_optimal_rpc_endpoint(self) -> str:
        """Get the optimal RPC endpoint based on latency and cost"""
        if not self.active_rpc_nodes:
            return self.config.backup_rpc_urls[0]
            
        # Return the fastest responding node
        return min(
            self.active_rpc_nodes.items(),
            key=lambda x: x[1].get('latency', float('inf'))
        )[0]
        
    async def calculate_monthly_overhead(self) -> Dict[str, float]:
        """Calculate and optimize monthly overhead costs"""
        current_costs = {
            'rpc_nodes': len(self.active_rpc_nodes) * self.config.rpc_cost_per_node,
            'infrastructure': self._calculate_infrastructure_cost(),
            'total': 0.0
        }
        current_costs['total'] = sum(current_costs.values())
        
        # Optimize if over budget
        if current_costs['total'] > self.config.max_monthly_budget:
            await self._reduce_costs(current_costs['total'])
            
        return current_costs
        
    async def _evaluate_scaling_needs(self):
        """Evaluate if infrastructure needs scaling"""
        portfolio_growth = (
            self.portfolio_size - self.config.min_portfolio_size
        ) / self.config.min_portfolio_size
        
        if portfolio_growth > self.config.scaling_threshold:
            # Calculate optimal number of RPC nodes
            optimal_nodes = min(
                self.config.max_rpc_nodes,
                1 + int(portfolio_growth / self.config.scaling_threshold)
            )
            
            # Scale up if needed and within budget
            current_costs = await self.calculate_monthly_overhead()
            potential_cost = current_costs['total'] + (
                (optimal_nodes - len(self.active_rpc_nodes)) * 
                self.config.rpc_cost_per_node
            )
            
            if potential_cost <= self.config.max_monthly_budget:
                await self._scale_rpc_nodes(optimal_nodes)
                
        self.last_scaling_check = datetime.now()
        
    async def _optimize_rpc_nodes(self):
        """Optimize RPC node allocation and usage"""
        for url in list(self.active_rpc_nodes.keys()):
            node_metrics = self.active_rpc_nodes[url]
            
            # Remove underperforming nodes
            if (node_metrics['errors'] > 100 or 
                node_metrics['latency'] > 1000):  # 1 second latency threshold
                await self._remove_rpc_node(url)
                
        # Add new nodes if needed and within budget
        if (len(self.active_rpc_nodes) < self.config.min_rpc_nodes and
            self._can_add_node()):
            await self._add_rpc_node()
            
    async def _update_resource_metrics(self):
        """Update system resource usage metrics"""
        # This would be implemented to get actual system metrics
        # For now, using placeholder values
        self.resource_metrics.update({
            'cpu_usage': 0.5,  # 50% CPU usage
            'memory_usage': 0.4,  # 40% memory usage
            'network_latency': 100,  # 100ms latency
            'monthly_cost': await self._calculate_total_cost()
        })
        
    def _should_check_scaling(self) -> bool:
        """Determine if it's time to check scaling needs"""
        return (datetime.now() - self.last_scaling_check) > timedelta(hours=24)
        
    async def _scale_rpc_nodes(self, target_nodes: int):
        """Scale RPC nodes to target number"""
        current_nodes = len(self.active_rpc_nodes)
        
        if current_nodes < target_nodes:
            # Scale up
            for _ in range(target_nodes - current_nodes):
                if self._can_add_node():
                    await self._add_rpc_node()
        elif current_nodes > target_nodes:
            # Scale down
            nodes_to_remove = current_nodes - target_nodes
            sorted_nodes = sorted(
                self.active_rpc_nodes.items(),
                key=lambda x: (x[1]['performance'], -x[1]['cost'])
            )
            
            for url, _ in sorted_nodes[:nodes_to_remove]:
                await self._remove_rpc_node(url)
                
    async def _add_rpc_node(self):
        """Add a new RPC node to the pool"""
        for url in self.config.backup_rpc_urls:
            if url not in self.active_rpc_nodes:
                self.active_rpc_nodes[url] = {
                    'latency': 0,
                    'errors': 0,
                    'performance': 1.0,
                    'cost': self.config.rpc_cost_per_node,
                    'added_date': datetime.now()
                }
                logger.info(f"Added new RPC node: {url}")
                break
                
    async def _remove_rpc_node(self, url: str):
        """Remove an RPC node from the pool"""
        if url in self.active_rpc_nodes:
            del self.active_rpc_nodes[url]
            logger.info(f"Removed RPC node: {url}")
            
    def _can_add_node(self) -> bool:
        """Check if we can add another RPC node within budget"""
        current_cost = sum(node['cost'] for node in self.active_rpc_nodes.values())
        return (current_cost + self.config.rpc_cost_per_node <= 
                self.config.max_monthly_budget)
                
    async def _reduce_costs(self, current_cost: float):
        """Reduce costs to stay within budget"""
        if current_cost <= self.config.max_monthly_budget:
            return
            
        # Remove least efficient nodes first
        sorted_nodes = sorted(
            self.active_rpc_nodes.items(),
            key=lambda x: (x[1]['performance'] / x[1]['cost'])
        )
        
        for url, node in sorted_nodes:
            if current_cost <= self.config.max_monthly_budget:
                break
            await self._remove_rpc_node(url)
            current_cost -= node['cost']
            
    def _calculate_infrastructure_cost(self) -> float:
        """Calculate infrastructure costs"""
        # This would be implemented to calculate actual infrastructure costs
        # For now, return a conservative estimate
        return 5.0  # $5 per month for basic infrastructure
        
    async def _calculate_total_cost(self) -> float:
        """Calculate total monthly costs"""
        costs = await self.calculate_monthly_overhead()
        return costs['total'] 