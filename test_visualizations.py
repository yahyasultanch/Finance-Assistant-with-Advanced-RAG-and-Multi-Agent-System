"""Test script for visualizations."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.visualizations import FinancialCharts
from datetime import datetime, timedelta
import random

# Sample data generator
def generate_sample_data():
    """Generate sample data for testing."""
    
    # Generate price history
    base_price = 150
    history = []
    for i in range(200):
        date = (datetime.now() - timedelta(days=200-i)).strftime('%Y-%m-%d')
        open_price = base_price + random.uniform(-5, 5)
        close_price = open_price + random.uniform(-3, 3)
        high_price = max(open_price, close_price) + random.uniform(0, 2)
        low_price = min(open_price, close_price) - random.uniform(0, 2)
        volume = random.randint(1000000, 10000000)
        
        history.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    price_data = {
        'current_price': base_price,
        'history': history
    }
    
    # Generate news data
    sentiments = ['positive', 'neutral', 'negative']
    news_data = []
    for i in range(30):
        date = (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
        news_data.append({
            'date': date,
            'title': f'News headline {i+1}',
            'sentiment': random.choice(sentiments),
            'score': random.uniform(0.5, 1.0)
        })
    
    # Generate predictions
    predictions = {
        'predictions': {
            'short_term': {
                'target_price': base_price * 1.02,
                'confidence': 'LOW',
                'confidence_score': 0.35
            },
            'long_term': {
                'target_price': base_price * 1.15,
                'upside_target': base_price * 1.25,
                'downside_target': base_price * 1.05,
                'confidence': 'MEDIUM',
                'confidence_score': 0.65
            }
        }
    }
    
    return price_data, news_data, predictions


def test_charts():
    """Test all chart types."""
    print("🧪 Testing Financial Charts...")
    
    # Initialize charts
    charts = FinancialCharts()
    print("✅ FinancialCharts initialized")
    
    # Generate sample data
    price_data, news_data, predictions = generate_sample_data()
    print("✅ Sample data generated")
    
    # Test candlestick chart
    print("\n📊 Testing Candlestick Chart...")
    candlestick_fig = charts.create_candlestick_chart(
        price_data=price_data,
        ticker="TEST",
        predictions=predictions,
        show_indicators=True
    )
    print(f"✅ Candlestick chart created: {len(candlestick_fig.data)} traces")
    
    # Test sentiment timeline
    print("\n📊 Testing Sentiment Timeline...")
    sentiment_fig = charts.create_sentiment_timeline(
        news_data=news_data,
        ticker="TEST"
    )
    print(f"✅ Sentiment timeline created: {len(sentiment_fig.data)} traces")
    
    # Test confidence chart
    print("\n📊 Testing Confidence Chart...")
    conf_fig = charts.create_prediction_confidence_chart(
        predictions=predictions,
        ticker="TEST"
    )
    print(f"✅ Confidence chart created: {len(conf_fig.data)} indicators")
    
    # Test performance chart
    print("\n📊 Testing Performance Chart...")
    perf_fig = charts.create_performance_chart(
        price_data=price_data,
        ticker="TEST"
    )
    print(f"✅ Performance chart created: {len(perf_fig.data)} traces")
    
    print("\n🎉 All tests passed!")
    print("\n💡 Run 'python run.py' to see charts in the Streamlit UI")


if __name__ == "__main__":
    test_charts()
