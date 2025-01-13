import numpy as np
from datetime import datetime
from openai import OpenAI
import os

class MarketStrategy:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.current_capital = initial_capital
        self.position = None
        self.returns = []
        self.trades = []
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ['GROQ_API_KEY']
        )

    def analyze_market_conditions(self, indicators, probability):
        """Analyze market conditions using LLM"""
        prompt = f"""As a financial strategist, provide a comprehensive market analysis:

        Current Market Indicators:
        - Gold Price: ${indicators['XAU BGNL']}
        - VIX (Volatility): {indicators['VIX']}
        - US 10Y Treasury: {indicators['USGG10YR']}%
        - Global Bond Index: {indicators['LUMSTRUU']}
        - Market Index: {indicators['LMBITR']}
        
        Model Prediction: {probability:.1%} probability of market increase

        Please provide a detailed analysis covering:

        1. Market Overview:
        - Current market sentiment and key trends
        - Impact of volatility levels
        - Relationship between indicators

        2. Risk Assessment:
        - Primary market risks
        - Potential catalysts
        - Key levels to watch

        3. Trading Strategy:
        - Recommended position direction
        - Entry and exit points
        - Position sizing considerations
        - Risk management rules

        4. Technical Outlook:
        - Support and resistance levels
        - Volatility considerations
        - Market momentum analysis

        Provide specific actionable recommendations and explain your reasoning.
        """

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    def generate_signals(self, probability, indicators):
        """Generate sophisticated trading signals with detailed analysis"""
        analysis = self.analyze_market_conditions(indicators, probability)
        
        # Base signal structure
        signal = {
            'action': 'HOLD',
            'size': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'confidence': 'Low',
            'analysis': analysis,
            'market_context': {
                'vix_level': 'High' if indicators['VIX'] > 25 else 'Moderate' if indicators['VIX'] > 15 else 'Low',
                'interest_rate_environment': 'High' if indicators['USGG10YR'] > 4 else 'Moderate' if indicators['USGG10YR'] > 2 else 'Low',
                'gold_trend': 'Bullish' if indicators['XAU BGNL'] > 2000 else 'Neutral' if indicators['XAU BGNL'] > 1800 else 'Bearish'
            }
        }
        
        # Enhanced trading logic with market context
        if probability > 0.7:
            signal.update({
                'action': 'BUY',
                'size': self.calculate_position_size(probability),
                'stop_loss': -0.02,
                'take_profit': 0.05,
                'confidence': 'High',
                'rationale': 'Strong bullish signal with high confidence'
            })
        elif probability < 0.3:
            signal.update({
                'action': 'SELL',
                'size': self.calculate_position_size(1 - probability),
                'stop_loss': 0.02,
                'take_profit': -0.05,
                'confidence': 'High',
                'rationale': 'Strong bearish signal with high confidence'
            })
        elif 0.55 < probability < 0.7:
            signal.update({
                'action': 'BUY',
                'size': self.calculate_position_size(probability) * 0.5,
                'stop_loss': -0.015,
                'take_profit': 0.03,
                'confidence': 'Medium',
                'rationale': 'Moderate bullish signal with reduced position size'
            })
        elif 0.3 < probability < 0.45:
            signal.update({
                'action': 'SELL',
                'size': self.calculate_position_size(1 - probability) * 0.5,
                'stop_loss': 0.015,
                'take_profit': -0.03,
                'confidence': 'Medium',
                'rationale': 'Moderate bearish signal with reduced position size'
            })
        
        # Add market context to trade history
        self.add_trade({
            'date': datetime.now(),
            'action': signal['action'],
            'size': signal['size'],
            'probability': probability,
            'indicators': indicators,
            'market_context': signal['market_context'],
            'rationale': signal.get('rationale', 'No clear directional signal')
        })
            
        return signal

    def chat_with_advisor(self, user_message, context=None):
        """Investment advisor chatbot"""
        if context is None:
            context = {
                'capital': self.current_capital,
                'returns': sum(self.returns) if self.returns else 0,
                'active_trades': len([t for t in self.trades if t['action'] != 'HOLD'])
            }
            
        prompt = f"""As an investment advisor, please answer this question:

        Portfolio Context:
        - Capital: ${context['capital']:,.2f}
        - Active Trades: {context['active_trades']}

        Client Question: {user_message}

        Please provide a professional and helpful response."""

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    def calculate_position_size(self, probability):
        """Calculate position size based on confidence"""
        # Risk per trade (1% of capital)
        risk_per_trade = self.current_capital * 0.01
        
        # Adjust size based on conviction
        confidence = abs(probability - 0.5) * 2
        position_size = risk_per_trade * confidence * 10
        
        # Limit to 10% of capital
        max_position = self.current_capital * 0.1
        return min(position_size, max_position)

    def update_portfolio(self, pnl):
        """Update portfolio with new profit/loss"""
        self.current_capital += pnl
        return_value = pnl / self.capital
        self.returns.append(return_value)
        
        # Automatically update positions based on profit/loss
        if abs(return_value) > 0.02:  # 2% threshold
            self.position = None  # Close position on significant moves

    def get_portfolio_stats(self):
        """Get basic portfolio statistics"""
        return {
            'initial_capital': self.capital,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.capital) / self.capital,
            'num_trades': len([t for t in self.trades if t['action'] != 'HOLD'])
        }

    def add_trade(self, trade_info):
        """Add trade to history"""
        self.trades.append(trade_info) 

    def calculate_metrics(self):
        """Calculate basic performance metrics"""
        return {
            'Sharpe Ratio': 0.0,  # Placeholder
            'Max Drawdown': 0.0,  # Placeholder
            'Win Rate': 0.0       # Placeholder
        } 