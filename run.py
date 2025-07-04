import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricingStrategy(Enum):
    """Pricing strategy options"""
    DEMAND_BASED = "demand_based"
    COMPETITOR_BASED = "competitor_based"
    COST_PLUS = "cost_plus"
    DYNAMIC = "dynamic"
    SURGE = "surge"

@dataclass
class Product:
    """Product information"""
    id: str
    name: str
    category: str
    base_price: float
    cost: float
    min_price: float
    max_price: float
    elasticity: float = -1.5  # Price elasticity of demand

@dataclass
class MarketData:
    """Market condition data"""
    timestamp: datetime
    demand_score: float  # 0-1 scale
    competitor_avg_price: float
    inventory_level: int
    seasonality_factor: float  # 0-2 scale
    promotional_activity: bool
    weather_impact: float = 1.0  # 0-2 scale

@dataclass
class PricingRecommendation:
    """Pricing recommendation output"""
    product_id: str
    current_price: float
    recommended_price: float
    strategy_used: PricingStrategy
    confidence_score: float
    expected_demand_change: float
    expected_revenue_change: float
    reasoning: str

class DynamicPricingEngine:
    """
    Dynamic pricing recommendation engine that analyzes market conditions,
    demand patterns, and competitive landscape to suggest optimal prices.
    """
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.price_history: List[Dict] = []
        self.market_data_history: List[MarketData] = []
        self.pricing_rules: Dict = self._initialize_pricing_rules()
        
    def _initialize_pricing_rules(self) -> Dict:
        """Initialize default pricing rules"""
        return {
            'max_price_increase_pct': 0.20,  # Max 20% price increase
            'max_price_decrease_pct': 0.15,  # Max 15% price decrease
            'min_margin_pct': 0.15,  # Minimum 15% margin
            'inventory_threshold_low': 10,
            'inventory_threshold_high': 100,
            'demand_threshold_low': 0.3,
            'demand_threshold_high': 0.8,
            'competitor_price_buffer': 0.05  # 5% buffer from competitor prices
        }
    
    def add_product(self, product: Product) -> None:
        """Add a product to the pricing engine"""
        self.products[product.id] = product
        logger.info(f"Added product: {product.name} (ID: {product.id})")
    
    def update_market_data(self, market_data: MarketData) -> None:
        """Update market conditions"""
        self.market_data_history.append(market_data)
        # Keep only last 30 days of data
        cutoff_date = datetime.now() - timedelta(days=30)
        self.market_data_history = [
            md for md in self.market_data_history 
            if md.timestamp >= cutoff_date
        ]
    
    def calculate_demand_elasticity_impact(self, product: Product, 
                                         price_change_pct: float) -> float:
        """Calculate demand change based on price elasticity"""
        return product.elasticity * price_change_pct
    
    def calculate_seasonality_adjustment(self, market_data: MarketData) -> float:
        """Calculate price adjustment based on seasonality"""
        if market_data.seasonality_factor > 1.5:
            return 1.1  # 10% increase for high season
        elif market_data.seasonality_factor < 0.7:
            return 0.9  # 10% decrease for low season
        return 1.0
    
    def calculate_demand_based_price(self, product: Product, 
                                   market_data: MarketData) -> float:
        """Calculate price based on demand conditions"""
        base_adjustment = 1.0
        
        # High demand - increase price
        if market_data.demand_score > self.pricing_rules['demand_threshold_high']:
            base_adjustment = 1.0 + (market_data.demand_score - 0.8) * 0.5
        # Low demand - decrease price
        elif market_data.demand_score < self.pricing_rules['demand_threshold_low']:
            base_adjustment = 1.0 - (0.3 - market_data.demand_score) * 0.3
        
        # Inventory adjustment
        if market_data.inventory_level < self.pricing_rules['inventory_threshold_low']:
            base_adjustment *= 1.15  # Low inventory - increase price
        elif market_data.inventory_level > self.pricing_rules['inventory_threshold_high']:
            base_adjustment *= 0.95  # High inventory - decrease price
        
        # Seasonality adjustment
        base_adjustment *= self.calculate_seasonality_adjustment(market_data)
        
        # Weather impact
        base_adjustment *= market_data.weather_impact
        
        new_price = product.base_price * base_adjustment
        return self._apply_price_constraints(product, new_price)
    
    def calculate_competitor_based_price(self, product: Product, 
                                       market_data: MarketData) -> float:
        """Calculate price based on competitor pricing"""
        competitor_price = market_data.competitor_avg_price
        buffer = self.pricing_rules['competitor_price_buffer']
        
        # Position slightly below competitor if demand is low
        if market_data.demand_score < 0.5:
            new_price = competitor_price * (1 - buffer)
        # Position slightly above if demand is high
        else:
            new_price = competitor_price * (1 + buffer)
        
        return self._apply_price_constraints(product, new_price)
    
    def calculate_surge_price(self, product: Product, 
                            market_data: MarketData) -> float:
        """Calculate surge pricing based on real-time demand"""
        surge_multiplier = 1.0
        
        # Calculate surge based on demand and inventory
        if (market_data.demand_score > 0.8 and 
            market_data.inventory_level < self.pricing_rules['inventory_threshold_low']):
            surge_multiplier = 1.0 + (market_data.demand_score * 0.5)
        
        # Promotional activity reduces surge
        if market_data.promotional_activity:
            surge_multiplier *= 0.9
        
        new_price = product.base_price * surge_multiplier
        return self._apply_price_constraints(product, new_price)
    
    def _apply_price_constraints(self, product: Product, price: float) -> float:
        """Apply business rules and constraints to price"""
        # Ensure minimum margin
        min_price_for_margin = product.cost * (1 + self.pricing_rules['min_margin_pct'])
        price = max(price, min_price_for_margin)
        
        # Apply min/max price constraints
        price = max(price, product.min_price)
        price = min(price, product.max_price)
        
        # Apply maximum change constraints
        max_increase = product.base_price * (1 + self.pricing_rules['max_price_increase_pct'])
        max_decrease = product.base_price * (1 - self.pricing_rules['max_price_decrease_pct'])
        
        price = min(price, max_increase)
        price = max(price, max_decrease)
        
        return round(price, 2)
    
    def calculate_confidence_score(self, product: Product, 
                                 market_data: MarketData,
                                 strategy: PricingStrategy) -> float:
        """Calculate confidence score for pricing recommendation"""
        confidence = 0.5  # Base confidence
        
        # More data points increase confidence
        if len(self.market_data_history) > 10:
            confidence += 0.2
        
        # Stable demand patterns increase confidence
        if len(self.market_data_history) >= 5:
            recent_demand = [md.demand_score for md in self.market_data_history[-5:]]
            demand_std = np.std(recent_demand)
            if demand_std < 0.1:
                confidence += 0.2
        
        # Strategy-specific confidence adjustments
        if strategy == PricingStrategy.DEMAND_BASED:
            if 0.3 <= market_data.demand_score <= 0.8:
                confidence += 0.1
        elif strategy == PricingStrategy.COMPETITOR_BASED:
            if market_data.competitor_avg_price > 0:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def select_optimal_strategy(self, product: Product, 
                              market_data: MarketData) -> PricingStrategy:
        """Select the best pricing strategy based on market conditions"""
        # High competition - use competitor-based pricing
        if market_data.competitor_avg_price > 0:
            return PricingStrategy.COMPETITOR_BASED
        
        # High demand volatility - use surge pricing
        if (market_data.demand_score > 0.8 and 
            market_data.inventory_level < self.pricing_rules['inventory_threshold_low']):
            return PricingStrategy.SURGE
        
        # Default to demand-based pricing
        return PricingStrategy.DEMAND_BASED
    
    def generate_recommendation(self, product_id: str, 
                              market_data: MarketData) -> PricingRecommendation:
        """Generate pricing recommendation for a product"""
        if product_id not in self.products:
            raise ValueError(f"Product {product_id} not found")
        
        product = self.products[product_id]
        current_price = product.base_price
        
        # Select optimal strategy
        strategy = self.select_optimal_strategy(product, market_data)
        
        # Calculate recommended price based on strategy
        if strategy == PricingStrategy.DEMAND_BASED:
            recommended_price = self.calculate_demand_based_price(product, market_data)
        elif strategy == PricingStrategy.COMPETITOR_BASED:
            recommended_price = self.calculate_competitor_based_price(product, market_data)
        elif strategy == PricingStrategy.SURGE:
            recommended_price = self.calculate_surge_price(product, market_data)
        else:
            recommended_price = current_price
        
        # Calculate expected impact
        price_change_pct = (recommended_price - current_price) / current_price
        expected_demand_change = self.calculate_demand_elasticity_impact(product, price_change_pct)
        expected_revenue_change = price_change_pct + expected_demand_change
        
        # Generate reasoning
        reasoning = self._generate_reasoning(product, market_data, strategy, 
                                           price_change_pct, expected_demand_change)
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score(product, market_data, strategy)
        
        return PricingRecommendation(
            product_id=product_id,
            current_price=current_price,
            recommended_price=recommended_price,
            strategy_used=strategy,
            confidence_score=confidence,
            expected_demand_change=expected_demand_change,
            expected_revenue_change=expected_revenue_change,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, product: Product, market_data: MarketData,
                          strategy: PricingStrategy, price_change_pct: float,
                          expected_demand_change: float) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasoning_parts = []
        
        # Strategy explanation
        if strategy == PricingStrategy.DEMAND_BASED:
            reasoning_parts.append(f"Using demand-based pricing due to current demand score of {market_data.demand_score:.2f}")
        elif strategy == PricingStrategy.COMPETITOR_BASED:
            reasoning_parts.append(f"Using competitor-based pricing with average competitor price at ${market_data.competitor_avg_price:.2f}")
        elif strategy == PricingStrategy.SURGE:
            reasoning_parts.append("Using surge pricing due to high demand and low inventory")
        
        # Market conditions
        if market_data.demand_score > 0.8:
            reasoning_parts.append("High demand detected")
        elif market_data.demand_score < 0.3:
            reasoning_parts.append("Low demand detected")
        
        if market_data.inventory_level < self.pricing_rules['inventory_threshold_low']:
            reasoning_parts.append("Low inventory levels")
        elif market_data.inventory_level > self.pricing_rules['inventory_threshold_high']:
            reasoning_parts.append("High inventory levels")
        
        # Price change impact
        if abs(price_change_pct) > 0.05:
            direction = "increase" if price_change_pct > 0 else "decrease"
            reasoning_parts.append(f"Recommending {price_change_pct:.1%} price {direction}")
        
        return ". ".join(reasoning_parts)
    
    def batch_recommendations(self, product_ids: List[str], 
                            market_data: MarketData) -> List[PricingRecommendation]:
        """Generate recommendations for multiple products"""
        recommendations = []
        for product_id in product_ids:
            try:
                recommendation = self.generate_recommendation(product_id, market_data)
                recommendations.append(recommendation)
            except ValueError as e:
                logger.error(f"Error generating recommendation for {product_id}: {e}")
        return recommendations
    
    def export_recommendations(self, recommendations: List[PricingRecommendation], 
                             filename: str) -> None:
        """Export recommendations to CSV"""
        data = []
        for rec in recommendations:
            data.append({
                'product_id': rec.product_id,
                'current_price': rec.current_price,
                'recommended_price': rec.recommended_price,
                'price_change_pct': (rec.recommended_price - rec.current_price) / rec.current_price,
                'strategy': rec.strategy_used.value,
                'confidence_score': rec.confidence_score,
                'expected_demand_change': rec.expected_demand_change,
                'expected_revenue_change': rec.expected_revenue_change,
                'reasoning': rec.reasoning
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Recommendations exported to {filename}")

# Example usage and testing
def main():
    """Example usage of the dynamic pricing engine"""
    
    # Initialize the pricing engine
    engine = DynamicPricingEngine()
    
    # Add sample products
    products = [
        Product(
            id="PROD_001",
            name="Wireless Headphones",
            category="Electronics",
            base_price=99.99,
            cost=50.00,
            min_price=79.99,
            max_price=149.99,
            elasticity=-1.2
        ),
        Product(
            id="PROD_002",
            name="Smart Phone Case",
            category="Electronics",
            base_price=29.99,
            cost=15.00,
            min_price=19.99,
            max_price=49.99,
            elasticity=-2.0
        ),
        Product(
            id="PROD_003",
            name="Coffee Mug",
            category="Home",
            base_price=12.99,
            cost=6.00,
            min_price=9.99,
            max_price=19.99,
            elasticity=-0.8
        )
    ]
    
    for product in products:
        engine.add_product(product)
    
    # Sample market data scenarios
    scenarios = [
        MarketData(
            timestamp=datetime.now(),
            demand_score=0.9,  # High demand
            competitor_avg_price=95.00,
            inventory_level=5,  # Low inventory
            seasonality_factor=1.2,
            promotional_activity=False,
            weather_impact=1.0
        ),
        MarketData(
            timestamp=datetime.now(),
            demand_score=0.3,  # Low demand
            competitor_avg_price=105.00,
            inventory_level=150,  # High inventory
            seasonality_factor=0.8,
            promotional_activity=True,
            weather_impact=0.9
        )
    ]
    
    print("=== Dynamic Pricing Recommendations ===\n")
    
    for i, market_data in enumerate(scenarios, 1):
        print(f"Scenario {i}:")
        print(f"  Demand Score: {market_data.demand_score}")
        print(f"  Inventory Level: {market_data.inventory_level}")
        print(f"  Competitor Price: ${market_data.competitor_avg_price}")
        print(f"  Seasonal Factor: {market_data.seasonality_factor}")
        print()
        
        # Generate recommendations
        recommendations = engine.batch_recommendations(
            [p.id for p in products], market_data
        )
        
        for rec in recommendations:
            product_name = engine.products[rec.product_id].name
            price_change = ((rec.recommended_price - rec.current_price) / rec.current_price) * 100
            
            print(f"  {product_name} ({rec.product_id}):")
            print(f"    Current Price: ${rec.current_price:.2f}")
            print(f"    Recommended: ${rec.recommended_price:.2f} ({price_change:+.1f}%)")
            print(f"    Strategy: {rec.strategy_used.value}")
            print(f"    Confidence: {rec.confidence_score:.2f}")
            print(f"    Expected Revenue Change: {rec.expected_revenue_change:.1%}")
            print(f"    Reasoning: {rec.reasoning}")
            print()
        
        print("-" * 50)

if __name__ == "__main__":
    main()
