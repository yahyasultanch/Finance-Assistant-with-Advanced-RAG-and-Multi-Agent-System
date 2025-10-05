"""Visualization utilities for financial data using Plotly."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FinancialCharts:
    """Generate interactive financial charts using Plotly."""
    
    def __init__(self):
        """Initialize chart configurations."""
        # Color scheme
        self.colors = {
            'up': '#00ff88',      # Green for bullish
            'down': '#ff4444',    # Red for bearish
            'neutral': '#ffaa00', # Orange for neutral
            'volume': '#5470c6',  # Blue for volume
            'line': '#3498db',    # Blue for line chart
            'prediction': '#9b59b6', # Purple for predictions
            'prediction_range': 'rgba(155, 89, 182, 0.2)', # Light purple for range
            'bg': '#0e1117',      # Dark background
            'grid': '#262730'     # Grid lines
        }
        
        # Default layout settings
        self.default_layout = {
            'template': 'plotly_dark',
            'plot_bgcolor': self.colors['bg'],
            'paper_bgcolor': self.colors['bg'],
            'font': {'family': 'Arial', 'size': 12, 'color': '#ffffff'},
            'hovermode': 'x unified',
            'margin': {'l': 60, 'r': 20, 't': 60, 'b': 40}
        }
    
    def create_main_price_chart(
        self,
        price_data: Dict[str, Any],
        ticker: str,
        predictions: Optional[Dict[str, Any]] = None,
        chart_type: str = "candlestick",
        time_range: str = "1Y"
    ) -> go.Figure:
        """
        Create main price chart with historical data and future predictions.
        
        Args:
            price_data: Price history data
            ticker: Stock ticker symbol
            predictions: Prediction results (optional)
            chart_type: "candlestick" or "line"
            time_range: "1W", "1M", "3M", "6M", "1Y", "ALL"
            
        Returns:
            Plotly Figure object
        """
        try:
            history = price_data.get('history', [])
            if not history:
                return self._empty_chart("No price data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.sort_values('Date')
            
            # Filter by time range
            end_date = df['Date'].max()
            if time_range == "1W":
                start_date = end_date - timedelta(days=7)
            elif time_range == "1M":
                start_date = end_date - timedelta(days=30)
            elif time_range == "3M":
                start_date = end_date - timedelta(days=90)
            elif time_range == "6M":
                start_date = end_date - timedelta(days=180)
            elif time_range == "1Y":
                start_date = end_date - timedelta(days=365)
            else:  # ALL
                start_date = df['Date'].min()
            
            df_filtered = df[df['Date'] >= start_date].copy()
            
            # Create figure with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} - Historical & Predicted Price', 'Volume')
            )
            
            current_price = price_data.get('current_price', df_filtered['close'].iloc[-1])
            last_date = df_filtered['Date'].iloc[-1]
            
            # Add historical price data
            if chart_type == "candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=df_filtered['Date'],
                        open=df_filtered['open'],
                        high=df_filtered['high'],
                        low=df_filtered['low'],
                        close=df_filtered['close'],
                        name='Historical Price',
                        increasing_line_color=self.colors['up'],
                        decreasing_line_color=self.colors['down']
                    ),
                    row=1, col=1
                )
            else:  # line chart
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['Date'],
                        y=df_filtered['close'],
                        name='Historical Price',
                        mode='lines',
                        line=dict(color=self.colors['line'], width=2),
                        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Add predictions if available
            if predictions:
                preds = predictions.get('predictions', {})
                
                # Short-term prediction (7 days)
                short_term = preds.get('short_term', {})
                if short_term:
                    st_target = short_term.get('target_price', 0)
                    st_date = last_date + timedelta(days=7)
                    
                    # Dotted line from current to short-term target
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, st_date],
                            y=[current_price, st_target],
                            name='7-Day Prediction',
                            mode='lines+markers',
                            line=dict(
                                color=self.colors['prediction'],
                                width=3,
                                dash='dot'
                            ),
                            marker=dict(size=10, symbol='diamond'),
                            hovertemplate='<b>7-Day Target</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Add confidence label
                    confidence = short_term.get('confidence', 'LOW')
                    fig.add_annotation(
                        x=st_date,
                        y=st_target,
                        text=f"7D: ${st_target:.2f} ({confidence})",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=self.colors['prediction'],
                        font=dict(size=11, color='white'),
                        bgcolor='rgba(155, 89, 182, 0.8)',
                        bordercolor=self.colors['prediction'],
                        borderwidth=2,
                        row=1, col=1
                    )
                
                # Long-term prediction (12 months)
                long_term = preds.get('long_term', {})
                if long_term:
                    lt_target = long_term.get('target_price', 0)
                    lt_upside = long_term.get('upside_target', 0)
                    lt_downside = long_term.get('downside_target', 0)
                    lt_date = last_date + timedelta(days=365)
                    
                    # Main prediction line
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, lt_date],
                            y=[current_price, lt_target],
                            name='12-Month Prediction',
                            mode='lines+markers',
                            line=dict(
                                color=self.colors['prediction'],
                                width=3,
                                dash='dash'
                            ),
                            marker=dict(size=12, symbol='star'),
                            hovertemplate='<b>12-Month Target</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Risk range (upside/downside)
                    if lt_upside and lt_downside:
                        fig.add_trace(
                            go.Scatter(
                                x=[lt_date, lt_date, lt_date],
                                y=[lt_downside, lt_target, lt_upside],
                                name='Risk Range',
                                mode='lines',
                                line=dict(color=self.colors['prediction'], width=0),
                                fill='toself',
                                fillcolor=self.colors['prediction_range'],
                                hovertemplate='<b>Risk Range</b><br>Downside: $%{y:.2f}<extra></extra>',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    
                    # Add confidence label
                    confidence = long_term.get('confidence', 'MEDIUM')
                    catalysts = long_term.get('catalysts_identified', 0)
                    fig.add_annotation(
                        x=lt_date,
                        y=lt_target,
                        text=f"12M: ${lt_target:.2f} ({confidence})<br>{catalysts} Catalysts",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=self.colors['prediction'],
                        font=dict(size=11, color='white'),
                        bgcolor='rgba(155, 89, 182, 0.8)',
                        bordercolor=self.colors['prediction'],
                        borderwidth=2,
                        row=1, col=1
                    )
            
            # Volume bars
            colors_volume = [
                self.colors['up'] if row['close'] >= row['open'] else self.colors['down']
                for _, row in df_filtered.iterrows()
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df_filtered['Date'],
                    y=df_filtered['volume'],
                    name='Volume',
                    marker_color=colors_volume,
                    opacity=0.5,
                    hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=dict(
                    text=f'{ticker} - Price Chart with AI Predictions ({time_range})',
                    font=dict(size=20)
                ),
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            # Disable the built-in range selector (we have our own in Streamlit)
            fig.update_xaxes(
                rangeslider=dict(visible=False),
                row=1, col=1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating main price chart: {e}")
            return self._empty_chart(f"Error: {str(e)}")
    
    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(**self.default_layout, height=400)
        return fig

    
    def create_candlestick_chart(
        self,
        price_data: Dict[str, Any],
        ticker: str,
        predictions: Optional[Dict[str, Any]] = None,
        show_indicators: bool = True
    ) -> go.Figure:
        """
        Create interactive candlestick chart with technical indicators.
        
        Args:
            price_data: Price history data
            ticker: Stock ticker symbol
            predictions: Prediction results (optional)
            show_indicators: Show moving averages and indicators
            
        Returns:
            Plotly Figure object
        """
        try:
            history = price_data.get('history', [])
            if not history:
                return self._empty_chart("No price data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.sort_values('Date')
            
            # Create figure with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} Price Action', 'Volume')
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['Date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    increasing_line_color=self.colors['up'],
                    decreasing_line_color=self.colors['down']
                ),
                row=1, col=1
            )
            
            # Add moving averages if requested
            if show_indicators and len(df) >= 50:
                # 50-day MA
                df['MA50'] = df['close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['MA50'],
                        name='50-Day MA',
                        line=dict(color=self.colors['ma_50'], width=1.5),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # 200-day MA if enough data
                if len(df) >= 200:
                    df['MA200'] = df['close'].rolling(window=200).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['MA200'],
                            name='200-Day MA',
                            line=dict(color=self.colors['ma_200'], width=1.5),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
            
            # Add price targets from predictions
            if predictions:
                current_price = price_data.get('current_price', 0)
                last_date = df['Date'].iloc[-1]
                
                # Short-term target (7 days ahead)
                short_term = predictions.get('predictions', {}).get('short_term', {})
                if short_term:
                    target_price = short_term.get('target_price', 0)
                    target_date = last_date + timedelta(days=7)
                    
                    # Add target line
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, target_date],
                            y=[current_price, target_price],
                            name='7-Day Target',
                            line=dict(color=self.colors['target'], width=2, dash='dash'),
                            mode='lines+markers',
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )
                
                # Long-term target (12 months ahead)
                long_term = predictions.get('predictions', {}).get('long_term', {})
                if long_term:
                    target_price = long_term.get('target_price', 0)
                    upside = long_term.get('upside_target', 0)
                    downside = long_term.get('downside_target', 0)
                    target_date = last_date + timedelta(days=365)
                    
                    # Add target line
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, target_date],
                            y=[current_price, target_price],
                            name='12-Month Target',
                            line=dict(color=self.colors['ma_200'], width=2, dash='dot'),
                            mode='lines+markers',
                            marker=dict(size=10, symbol='star')
                        ),
                        row=1, col=1
                    )
                    
                    # Add risk range
                    if upside and downside:
                        fig.add_trace(
                            go.Scatter(
                                x=[target_date, target_date],
                                y=[downside, upside],
                                name='Risk Range',
                                mode='lines',
                                line=dict(color=self.colors['target'], width=8),
                                opacity=0.3,
                                showlegend=True
                            ),
                            row=1, col=1
                        )
            
            # Volume bars
            colors_volume = [
                self.colors['up'] if row['close'] >= row['open'] else self.colors['down']
                for _, row in df.iterrows()
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df['Date'],
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors_volume,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=f'{ticker} - Candlestick Chart with Predictions',
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return self._empty_chart(f"Error: {str(e)}")
    
    def create_sentiment_timeline(
        self,
        news_data: List[Dict[str, Any]],
        ticker: str
    ) -> go.Figure:
        """
        Create sentiment evolution timeline chart.
        
        Args:
            news_data: List of news items with sentiment scores
            ticker: Stock ticker symbol
            
        Returns:
            Plotly Figure object
        """
        try:
            if not news_data:
                return self._empty_chart("No sentiment data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(news_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate rolling average sentiment
            df['sentiment_numeric'] = df['sentiment'].map({
                'positive': 1,
                'neutral': 0,
                'negative': -1
            })
            df['sentiment_ma'] = df['sentiment_numeric'].rolling(window=5, min_periods=1).mean()
            
            # Create figure
            fig = go.Figure()
            
            # Add sentiment points
            for sentiment_type in ['positive', 'neutral', 'negative']:
                subset = df[df['sentiment'] == sentiment_type]
                fig.add_trace(
                    go.Scatter(
                        x=subset['date'],
                        y=subset['sentiment_numeric'],
                        name=sentiment_type.capitalize(),
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=self.colors.get(sentiment_type, self.colors['neutral']),
                            symbol='circle',
                            line=dict(width=1, color='white')
                        ),
                        text=subset['title'],
                        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Sentiment: ' + sentiment_type
                    )
                )
            
            # Add moving average line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sentiment_ma'],
                    name='5-Day MA',
                    mode='lines',
                    line=dict(color='#3498db', width=3),
                    hovertemplate='<b>Sentiment MA</b><br>%{y:.2f}'
                )
            )
            
            # Add horizontal zones
            fig.add_hrect(y0=0.3, y1=1, fillcolor=self.colors['positive'], opacity=0.1, line_width=0)
            fig.add_hrect(y0=-0.3, y1=0.3, fillcolor=self.colors['neutral'], opacity=0.1, line_width=0)
            fig.add_hrect(y0=-1, y1=-0.3, fillcolor=self.colors['negative'], opacity=0.1, line_width=0)
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=f'{ticker} - Sentiment Evolution Over Time',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                height=500,
                yaxis=dict(
                    range=[-1.2, 1.2],
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment timeline: {e}")
            return self._empty_chart(f"Error: {str(e)}")
    
    def create_prediction_confidence_chart(
        self,
        predictions: Dict[str, Any],
        ticker: str
    ) -> go.Figure:
        """
        Create gauge charts for prediction confidence levels.
        
        Args:
            predictions: Prediction results
            ticker: Stock ticker symbol
            
        Returns:
            Plotly Figure object
        """
        try:
            preds = predictions.get('predictions', {})
            short_term = preds.get('short_term', {})
            long_term = preds.get('long_term', {})
            
            # Create subplots for gauges
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('Short-Term (1-7 Days)', 'Long-Term (3-12 Months)')
            )
            
            # Short-term gauge
            short_conf = short_term.get('confidence_score', 0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=short_conf,
                    title={'text': f"Confidence: {short_term.get('confidence', 'LOW')}"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_confidence_color(short_conf)},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(255, 68, 68, 0.3)'},
                            {'range': [30, 50], 'color': 'rgba(255, 170, 0, 0.3)'},
                            {'range': [50, 70], 'color': 'rgba(255, 221, 87, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(38, 194, 129, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Long-term gauge
            long_conf = long_term.get('confidence_score', 0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=long_conf,
                    title={'text': f"Confidence: {long_term.get('confidence', 'LOW')}"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_confidence_color(long_conf)},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(255, 68, 68, 0.3)'},
                            {'range': [30, 50], 'color': 'rgba(255, 170, 0, 0.3)'},
                            {'range': [50, 70], 'color': 'rgba(255, 221, 87, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(38, 194, 129, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=f'{ticker} - Prediction Confidence Levels',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence chart: {e}")
            return self._empty_chart(f"Error: {str(e)}")
    
    def create_performance_chart(
        self,
        price_data: Dict[str, Any],
        ticker: str
    ) -> go.Figure:
        """
        Create YTD performance chart.
        
        Args:
            price_data: Price history data
            ticker: Stock ticker symbol
            
        Returns:
            Plotly Figure object
        """
        try:
            history = price_data.get('history', [])
            if not history:
                return self._empty_chart("No price data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.sort_values('Date')
            
            # Calculate cumulative returns
            first_price = df['close'].iloc[0]
            df['return_pct'] = ((df['close'] - first_price) / first_price) * 100
            
            # Create figure
            fig = go.Figure()
            
            # Add performance line
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['return_pct'],
                    name='Return %',
                    mode='lines',
                    line=dict(color=self.colors['ma_50'], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(250, 200, 88, 0.2)'
                )
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=f'{ticker} - Cumulative Return (%)',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return self._empty_chart(f"Error: {str(e)}")
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 70:
            return self.colors['positive']
        elif confidence >= 50:
            return self.colors['neutral']
        else:
            return self.colors['negative']
    
    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(**self.default_layout, height=400)
        return fig
