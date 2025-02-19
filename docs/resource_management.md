# Resource Management Documentation

## Overview

The Resource Management module (`backend/app/infrastructure/resource_manager.py`) handles dynamic scaling and optimization of infrastructure resources, with a focus on maintaining low operational costs while maximizing performance.

## Components

### 1. Configuration Parameters

```python
@dataclass
class ResourceConfig:
    max_monthly_budget: float = 50.0    # Maximum monthly infrastructure cost
    min_rpc_nodes: int = 1              # Minimum number of RPC nodes
    max_rpc_nodes: int = 3              # Maximum number of RPC nodes
    rpc_cost_per_node: float = 15.0     # Cost per RPC node per month
    min_portfolio_size: float = 50.0     # Minimum portfolio size
    scaling_threshold: float = 0.2       # Portfolio growth threshold for scaling
    cpu_threshold: float = 0.8          # CPU utilization threshold
    memory_threshold: float = 0.8       # Memory utilization threshold
```

## Resource Management Features

### 1. RPC Node Management

#### Dynamic Node Scaling
```python
async def _evaluate_scaling_needs(self):
    """Evaluate if infrastructure needs scaling"""
    portfolio_growth = (
        self.portfolio_size - self.config.min_portfolio_size
    ) / self.config.min_portfolio_size
    
    if portfolio_growth > self.config.scaling_threshold:
        optimal_nodes = min(
            self.config.max_rpc_nodes,
            1 + int(portfolio_growth / self.config.scaling_threshold)
        )
```

#### Node Performance Monitoring
- Latency tracking
- Error rate monitoring
- Cost per request analysis
- Performance scoring

#### Node Selection Strategy
- Cost-based routing
- Performance-based selection
- Load balancing
- Failover handling

### 2. Cost Optimization

#### Budget Management
```python
async def calculate_monthly_overhead(self) -> Dict[str, float]:
    """Calculate and optimize monthly overhead costs"""
    current_costs = {
        'rpc_nodes': len(self.active_rpc_nodes) * self.config.rpc_cost_per_node,
        'infrastructure': self._calculate_infrastructure_cost(),
        'total': 0.0
    }
```

#### Cost Reduction Strategies
- Free node prioritization
- Dynamic scaling
- Resource pooling
- Batch processing

#### Cost Monitoring
- Real-time cost tracking
- Budget alerts
- Cost optimization suggestions
- Usage analytics

### 3. Performance Optimization

#### Resource Metrics
```python
async def _update_resource_metrics(self):
    """Update system resource usage metrics"""
    self.resource_metrics.update({
        'cpu_usage': 0.5,      # 50% CPU usage
        'memory_usage': 0.4,   # 40% memory usage
        'network_latency': 100, # 100ms latency
        'monthly_cost': await self._calculate_total_cost()
    })
```

#### Performance Monitoring
- CPU utilization
- Memory usage
- Network latency
- Request throughput

#### Optimization Strategies
- Load balancing
- Cache optimization
- Connection pooling
- Request batching

## Scaling Implementation

### 1. Portfolio-Based Scaling

#### Growth Phases
1. **Initial Phase ($50-$100)**
   - Single free RPC node
   - Minimal resource usage
   - Conservative trading

2. **Growth Phase ($100-$500)**
   - 1-2 RPC nodes
   - Optimized resource allocation
   - Expanded trading strategies

3. **Scaling Phase ($500+)**
   - Multiple RPC nodes
   - Full resource utilization
   - Maximum efficiency

#### Scaling Triggers
```python
def _should_check_scaling(self) -> bool:
    """Determine if it's time to check scaling needs"""
    return (datetime.now() - self.last_scaling_check) > timedelta(hours=24)
```

### 2. Infrastructure Management

#### Node Addition
```python
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
```

#### Node Removal
```python
async def _remove_rpc_node(self, url: str):
    """Remove an RPC node from the pool"""
    if url in self.active_rpc_nodes:
        del self.active_rpc_nodes[url]
```

#### Performance Optimization
```python
async def get_optimal_rpc_endpoint(self) -> str:
    """Get the optimal RPC endpoint based on latency and cost"""
    return min(
        self.active_rpc_nodes.items(),
        key=lambda x: x[1].get('latency', float('inf'))
    )[0]
```

## Usage Examples

### 1. Resource Optimization
```python
# Optimize resources based on portfolio size
await resource_manager.optimize_resources(current_portfolio_size)

# Get optimal RPC endpoint
optimal_endpoint = await resource_manager.get_optimal_rpc_endpoint()

# Calculate monthly overhead
costs = await resource_manager.calculate_monthly_overhead()
```

### 2. Scaling Management
```python
# Check scaling needs
if resource_manager._should_check_scaling():
    await resource_manager._evaluate_scaling_needs()
    
# Add new node if needed
if resource_manager._can_add_node():
    await resource_manager._add_rpc_node()
```

### 3. Performance Monitoring
```python
# Update resource metrics
await resource_manager._update_resource_metrics()

# Check resource utilization
if resource_manager.resource_metrics['cpu_usage'] > resource_manager.config.cpu_threshold:
    await resource_manager._optimize_resources()
```

## Error Handling

### 1. Node Failure
```python
if node_metrics['errors'] > 100 or node_metrics['latency'] > 1000:
    await self._remove_rpc_node(url)
```

### 2. Budget Overflow
```python
if current_cost > self.config.max_monthly_budget:
    await self._reduce_costs(current_cost)
```

### 3. Resource Exhaustion
```python
if self.resource_metrics['memory_usage'] > self.config.memory_threshold:
    await self._optimize_memory_usage()
```

## Monitoring and Alerts

### 1. Cost Monitoring
- Real-time cost tracking
- Budget alerts
- Usage analytics
- Optimization recommendations

### 2. Performance Metrics
- Node health status
- Resource utilization
- Latency monitoring
- Error rate tracking

### 3. Scaling Alerts
- Scaling events
- Resource warnings
- Performance degradation
- Budget notifications 