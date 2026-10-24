"""Price chart with predictions, using Plotly."""
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

RANGES = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}


class FinancialCharts:
    """Generate interactive financial charts using Plotly."""

    colors = {
        "up": "#00c805",
        "down": "#ff5000",
        "line": "#3498db",
        "prediction": "#b37feb",
        "band": "rgba(179, 127, 235, 0.12)",
        "bg": "#0e1117",
    }

    def create_main_price_chart(
        self,
        price_data: Dict[str, Any],
        ticker: str,
        predictions: Optional[Dict[str, Any]] = None,
        chart_type: str = "candlestick",
        time_range: str = "1Y",
        show_long_term: bool = True,
    ) -> go.Figure:
        """
        Price history (candles or line) and volume, with the 7-day forecast
        drawn forward from the last close and the 12-month target as a
        horizontal level with its risk band. The x-axis stays on the selected
        range (plus the 7 days ahead): drawing the 12-month path a year out
        would squeeze any range shorter than a year into a sliver.
        """
        history = price_data.get("history", [])
        if not history:
            return self._empty_chart("No price data available")

        df = pd.DataFrame(history)
        df["Date"] = pd.to_datetime(df["date"])
        df = df.sort_values("Date")
        if time_range in RANGES:
            df = df[df["Date"] >= df["Date"].max() - timedelta(days=RANGES[time_range])]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.75, 0.25])

        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=df["Date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name="Price", increasing_line_color=self.colors["up"],
                decreasing_line_color=self.colors["down"],
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["close"], name="Close", mode="lines",
                line=dict(color=self.colors["line"], width=2),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        last_date, last_close = df["Date"].iloc[-1], float(df["close"].iloc[-1])
        horizons = (predictions or {}).get("predictions", {})

        short_term = horizons.get("short_term", {})
        if short_term.get("target_price"):
            target, end = short_term["target_price"], last_date + timedelta(days=7)
            fig.add_trace(go.Scatter(
                x=[last_date, end], y=[last_close, target], name="7-day forecast",
                mode="lines+markers", line=dict(color=self.colors["prediction"], width=2, dash="dot"),
                marker=dict(size=[0, 9], symbol="diamond"),
                hovertemplate="7-day forecast<br>%{x|%b %d}: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            fig.add_annotation(
                x=end, y=target, text=f"7D ${target:.2f} ({short_term.get('confidence', 'low')})",
                showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(size=11, color="white"),
                bgcolor="rgba(179, 127, 235, 0.85)", row=1, col=1,
            )

        long_term = horizons.get("long_term", {})
        if show_long_term and long_term.get("target_price"):
            target = long_term["target_price"]
            low, high = long_term.get("downside_target"), long_term.get("upside_target")
            if low and high:
                fig.add_hrect(y0=low, y1=high, fillcolor=self.colors["band"], line_width=0,
                              row=1, col=1)
            fig.add_hline(
                y=target, line=dict(color=self.colors["prediction"], width=1.5, dash="dash"),
                annotation_text=(f"12M target ${target:.2f}"
                                 + (f" (range ${low:.0f}-${high:.0f})" if low and high else "")),
                annotation_position="top left", annotation_font_color=self.colors["prediction"],
                row=1, col=1,
            )

        fig.add_trace(go.Bar(
            x=df["Date"], y=df["volume"], name="Volume", showlegend=False, opacity=0.5,
            marker_color=[self.colors["up"] if c >= o else self.colors["down"]
                          for o, c in zip(df["open"], df["close"])],
            hovertemplate="Volume %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", plot_bgcolor=self.colors["bg"], paper_bgcolor=self.colors["bg"],
            height=650, hovermode="x unified", margin=dict(l=60, r=20, t=50, b=40),
            title=dict(text=f"{ticker} price ({time_range}) with forecasts", font=dict(size=18)),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
        )
        # Skip weekends so candles sit next to each other
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def _empty_chart(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color="gray"))
        fig.update_layout(template="plotly_dark", height=400,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
