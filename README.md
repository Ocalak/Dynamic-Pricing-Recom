# Dynamic Pricing Engine

A comprehensive Python library for dynamic pricing recommendations that adapts to market conditions, demand patterns, and competitive landscapes.

## Features

- **Multiple Pricing Strategies**: Demand-based, competitor-based, surge pricing
- **Market Factor Analysis**: Demand scoring, inventory levels, seasonality
- **Business Rules**: Price constraints, margin requirements, elasticity modeling
- **Confidence Scoring**: AI-powered recommendation confidence assessment
- **Batch Processing**: Handle multiple products simultaneously
- **Export Capabilities**: CSV export for analysis and reporting

## Installation

```bash
pip install dynamic-pricing-engine
```
```
git clone https://github.com/yourusername/dynamic-pricing-engine.git
cd dynamic-pricing-engine
pip install -e .
```

from dynamic_pricing_engine import DynamicPricingEngine, Product, MarketData
from datetime import datetime

# Initialize the engine
engine = DynamicPricingEngine()

# Add a product
product = Product(
    id="PROD_001",
    name="Wireless Headphones",
    category="Electronics",
    base_price=99.99,
    cost=50.00,
    min_price=79.99,
    max_price=149.99,
    elasticity=-1.2
)
engine.add_product(product)

# Create market data
market_data = MarketData(
    timestamp=datetime.now(),
    demand_score=0.8,
    competitor_avg_price=95.00,
    inventory_level=25,
    seasonality_factor=1.1,
    promotional_activity=False
)

# Generate recommendation
recommendation = engine.generate_recommendation("PROD_001", market_data)
print(f"Recommended price: ${recommendation.recommended_price:.2f}")

# Quick Start
pythonfrom dynamic_pricing_engine import DynamicPricingEngine, Product, MarketData
from datetime import datetime

# Initialize the engine
engine = DynamicPricingEngine()

# Add a product
product = Product(
    id="PROD_001",
    name="Wireless Headphones",
    category="Electronics",
    base_price=99.99,
    cost=50.00,
    min_price=79.99,
    max_price=149.99,
    elasticity=-1.2
)
engine.add_product(product)

# Create market data
market_data = MarketData(
    timestamp=datetime.now(),
    demand_score=0.8,
    competitor_avg_price=95.00,
    inventory_level=25,
    seasonality_factor=1.1,
    promotional_activity=False
)

# Testing
Run the test suite:
```
bashpytest tests/
```
