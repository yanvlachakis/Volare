from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json
from typing import Dict, Optional
import asyncio
from datetime import datetime

app = FastAPI()

class Dashboard:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.performance_data: Optional[Dict] = None
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast_performance(self, data: Dict):
        self.performance_data = data
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                continue

dashboard = Dashboard()

@app.get("/")
async def get():
    """Serve the dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Volare Trading Bot Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .metric-card {
                    @apply bg-white rounded-lg shadow-md p-4 m-2;
                }
                .metric-value {
                    @apply text-2xl font-bold text-blue-600;
                }
                .metric-label {
                    @apply text-sm text-gray-600;
                }
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-3xl font-bold mb-8">Volare Trading Bot Dashboard</h1>
                
                <!-- Portfolio Metrics -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                    <div class="metric-card">
                        <div class="metric-value" id="current-capital">$0.00</div>
                        <div class="metric-label">Current Capital</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="total-return">0.00%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="win-rate">0.00%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="profit-factor">0.00</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>
                
                <!-- Charts -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div id="equity-curve"></div>
                    </div>
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div id="strategy-performance"></div>
                    </div>
                </div>
                
                <!-- Resource Usage -->
                <div class="bg-white rounded-lg shadow-md p-4 mb-8">
                    <h2 class="text-xl font-bold mb-4">Resource Usage</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div id="cpu-gauge"></div>
                        <div id="memory-gauge"></div>
                        <div id="cost-gauge"></div>
                    </div>
                </div>
                
                <!-- Recent Alerts -->
                <div class="bg-white rounded-lg shadow-md p-4">
                    <h2 class="text-xl font-bold mb-4">Recent Alerts</h2>
                    <div id="alerts" class="space-y-2"></div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                let equityCurveData = [];
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // Update metrics
                    document.getElementById('current-capital').textContent = 
                        `$${data.portfolio.current_capital.toFixed(2)}`;
                    document.getElementById('total-return').textContent = 
                        `${(data.portfolio.total_return * 100).toFixed(2)}%`;
                    document.getElementById('win-rate').textContent = 
                        `${(data.trading.win_rate * 100).toFixed(2)}%`;
                    document.getElementById('profit-factor').textContent = 
                        data.trading.profit_factor.toFixed(2);
                    
                    // Update equity curve
                    equityCurveData.push({
                        x: new Date(),
                        y: data.portfolio.current_capital
                    });
                    
                    if (equityCurveData.length > 100) {
                        equityCurveData.shift();
                    }
                    
                    Plotly.newPlot('equity-curve', [{
                        x: equityCurveData.map(d => d.x),
                        y: equityCurveData.map(d => d.y),
                        type: 'scatter',
                        name: 'Equity'
                    }], {
                        title: 'Equity Curve',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Capital ($)' }
                    });
                    
                    // Update strategy performance
                    const strategies = Object.entries(data.strategies).map(([name, stats]) => ({
                        strategy: name,
                        winRate: stats.winning_trades / stats.total_trades,
                        pnl: stats.total_pnl
                    }));
                    
                    Plotly.newPlot('strategy-performance', [{
                        x: strategies.map(s => s.strategy),
                        y: strategies.map(s => s.pnl),
                        type: 'bar',
                        name: 'PnL'
                    }], {
                        title: 'Strategy Performance',
                        xaxis: { title: 'Strategy' },
                        yaxis: { title: 'PnL ($)' }
                    });
                    
                    // Update resource gauges
                    const resourceConfig = {
                        showscale: false,
                        domain: { x: [0, 1], y: [0, 1] }
                    };
                    
                    Plotly.newPlot('cpu-gauge', [{
                        type: "indicator",
                        mode: "gauge+number",
                        value: data.resource_usage.cpu * 100,
                        title: { text: "CPU Usage %" },
                        gauge: {
                            axis: { range: [0, 100] },
                            bar: { color: "darkblue" },
                            steps: [
                                { range: [0, 50], color: "lightgray" },
                                { range: [50, 80], color: "gray" }
                            ],
                            threshold: {
                                line: { color: "red", width: 4 },
                                thickness: 0.75,
                                value: 80
                            }
                        }
                    }], resourceConfig);
                    
                    Plotly.newPlot('memory-gauge', [{
                        type: "indicator",
                        mode: "gauge+number",
                        value: data.resource_usage.memory * 100,
                        title: { text: "Memory Usage %" },
                        gauge: {
                            axis: { range: [0, 100] },
                            bar: { color: "darkblue" },
                            steps: [
                                { range: [0, 50], color: "lightgray" },
                                { range: [50, 80], color: "gray" }
                            ],
                            threshold: {
                                line: { color: "red", width: 4 },
                                thickness: 0.75,
                                value: 80
                            }
                        }
                    }], resourceConfig);
                    
                    Plotly.newPlot('cost-gauge', [{
                        type: "indicator",
                        mode: "gauge+number",
                        value: data.resource_usage.cost,
                        title: { text: "Monthly Cost ($)" },
                        gauge: {
                            axis: { range: [0, 50] },
                            bar: { color: "darkblue" },
                            steps: [
                                { range: [0, 25], color: "lightgray" },
                                { range: [25, 40], color: "gray" }
                            ],
                            threshold: {
                                line: { color: "red", width: 4 },
                                thickness: 0.75,
                                value: 45
                            }
                        }
                    }], resourceConfig);
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await dashboard.connect(websocket)
    try:
        if dashboard.performance_data:
            await websocket.send_json(dashboard.performance_data)
        while True:
            await websocket.receive_text()
    except:
        dashboard.disconnect(websocket)

async def update_dashboard(performance_tracker):
    """Update dashboard with latest performance metrics"""
    while True:
        try:
            performance_data = performance_tracker.get_performance_summary()
            await dashboard.broadcast_performance(performance_data)
        except Exception as e:
            print(f"Dashboard update error: {str(e)}")
        await asyncio.sleep(1)  # Update every second 