"""Streamlit UI for AI Finance Assistant."""
import streamlit as st
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger(__name__)

from config import (
    DATA_DIR, INDEX_DIR, REPORTS_DIR,
    SEC_USER_AGENT, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT, AZURE_EMBEDDING_KEY,
    AZURE_OPENAI_API_VERSION, FINBERT_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
)
from agents.orchestrator import Orchestrator
from utils.visualizations import FinancialCharts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page config
st.set_page_config(
    page_title="AI Investment Research Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    """Initialize and cache orchestrator."""
    return Orchestrator(
        sec_user_agent=SEC_USER_AGENT,
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        reports_dir=REPORTS_DIR,
        embedding_model_name="text-embedding-ada-002",  # kept for compatibility
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_key=AZURE_OPENAI_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_embedding_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        azure_embedding_key=AZURE_EMBEDDING_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        finbert_model=FINBERT_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K_RETRIEVAL
    )


def main():
    """Main Streamlit app."""
    
    # Title and description
    st.title("📊 AI Investment Research Assistant")
    st.markdown("""
    Generate comprehensive analyst briefs using RAG, AI agents, and financial sentiment analysis.
    
    **Features:**
    - 📑 SEC filing analysis (10-K, 10-Q, 8-K)
    - 📰 News aggregation and sentiment
    - 🤖 RAG-powered summaries with citations
    - 💹 Price and performance metrics
    - 📈 FinBERT sentiment analysis
    """)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
        ).upper()
        
        use_cache = st.checkbox(
            "Use cached data",
            value=True,
            help="Use previously downloaded filings if available"
        )
        
        days_back = st.slider(
            "Days to look back for filings",
            min_value=90,
            max_value=730,
            value=365,
            step=30,
            help="Number of days to search for SEC filings"
        )
        
        save_format = st.selectbox(
            "Report format",
            options=["markdown", "html"],
            help="Output format for the report"
        )
        
        st.divider()
        
        st.markdown("### 🔧 System Info")
        st.info(f"""
        **Azure OpenAI Model:** {AZURE_OPENAI_DEPLOYMENT}
        **Embedding:** text-embedding-ada-002
        **FinBERT:** {FINBERT_MODEL.split('/')[-1]}
        """)
        
        # Generate brief button
        generate_button = st.button("🚀 Generate Brief", type="primary", use_container_width=True)
        
        if generate_button:
            # Store input params in session state
            st.session_state['ticker'] = ticker
            st.session_state['use_cache'] = use_cache
            st.session_state['days_back'] = days_back
            st.session_state['save_format'] = save_format
            st.session_state['generate_clicked'] = True
            st.session_state['result'] = None  # Clear previous result
        
    # Main content area - Check if we should display results
    if st.session_state.get('generate_clicked', False):
        # Get params from session state
        ticker = st.session_state.get('ticker', '')
        use_cache = st.session_state.get('use_cache', True)
        days_back = st.session_state.get('days_back', 365)
        save_format = st.session_state.get('save_format', 'markdown')
        
        if not ticker:
            st.error("Please enter a stock ticker")
            st.session_state['generate_clicked'] = False
            return
        
        # Check if we already have results
        if st.session_state.get('result') is None:
            try:
                orchestrator = get_orchestrator()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Ingestion
                status_text.text("📥 Step 1/5: Ingesting SEC filings and news...")
                progress_bar.progress(10)
                
                with st.spinner("Fetching data from SEC EDGAR and news sources..."):
                    # This will be done inside orchestrator
                    pass
                
                progress_bar.progress(30)
                
                # Step 2: Generate brief
                status_text.text("🤖 Step 2/5: Generating RAG summaries...")
                progress_bar.progress(40)
                
                with st.spinner("Running retrieval-augmented generation..."):
                    result = orchestrator.generate_brief(
                        ticker=ticker,
                        use_cache=use_cache,
                        days_back=days_back,
                        save_format=save_format
                    )
                
                # Store result in session state
                st.session_state['result'] = result
                
                progress_bar.progress(80)
                
                # Step 3: Display results
                status_text.text("📊 Step 3/5: Analyzing sentiment...")
                progress_bar.progress(90)
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
            except Exception as e:
                st.error(f"Error generating brief: {str(e)}")
                logger.error(f"Error in brief generation: {e}", exc_info=True)
                st.session_state['generate_clicked'] = False
                return
        
        # Get result from session state
        result = st.session_state.get('result')
        if result:
            # Display success message
            st.success(f"✅ Analyst brief generated for **{ticker}**!")
            
            # Display report
            st.divider()
            st.header(f"📄 Analyst Brief - {ticker}")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Report", "📈 Charts", "📊 Metrics", "💬 Sentiment", "🔍 Sources"])
            
            with tab1:
                st.markdown(result["report_content"])
                
                # Download button
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=result["report_content"],
                    file_name=f"{ticker}_analyst_brief.md",
                    mime="text/markdown"
                )
            
            with tab2:
                st.subheader("📈 Interactive Price Chart")
                
                # Initialize chart generator
                charts = FinancialCharts()
                
                # Get data
                price_data = result.get("price_data", {})
                prediction_results = result.get("prediction_results", {})
                
                # Chart controls
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    chart_type = st.radio(
                        "Chart Type",
                        options=["Candlestick", "Line"],
                        index=0,
                        help="Toggle between candlestick and line chart"
                    )
                
                with col2:
                    time_range = st.radio(
                        "Time Range",
                        options=["1W", "1M", "3M", "6M", "1Y", "ALL"],
                        index=4,  # Default to 1Y
                        horizontal=True,
                        help="Select historical time range"
                    )
                
                # Create and display main chart
                main_fig = charts.create_main_price_chart(
                    price_data=price_data,
                    ticker=ticker,
                    predictions=prediction_results,
                    chart_type=chart_type.lower(),
                    time_range=time_range
                )
                st.plotly_chart(main_fig, use_container_width=True, key="main_price_chart")
                
                # Legend explanation
                st.markdown("""
                **Chart Legend:**
                - **Solid Line/Candles**: Historical price data
                - **Dotted Purple Line**: Short-term prediction (7 days)
                - **Dashed Purple Line**: Long-term prediction (12 months)
                - **Shaded Area**: Risk-adjusted range (upside/downside)
                - **Volume Bars**: Trading volume (green = up day, red = down day)
                
                💡 **Tip**: Use the range selector buttons above the chart or the timeline controls to adjust the view.
                """)
            
            with tab3:
                st.subheader("Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                price_metrics = result.get("price_metrics", {})
                company_info = result.get("company_info", {})
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${price_metrics.get('current_price', 0):.2f}",
                        delta=f"{price_metrics.get('ytd_return', 0):.2f}% YTD"
                    )
                
                with col2:
                    market_cap = company_info.get("market_cap", 0)
                    if market_cap > 1e9:
                        st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                    else:
                        st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
                
                with col3:
                    st.metric(
                        "52-Week High",
                        f"${price_metrics.get('52_week_high', 0):.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Volatility",
                        f"{price_metrics.get('volatility', 0):.2f}%"
                    )
            
            with tab4:
                st.subheader("Sentiment Analysis")
                
                sentiment = result.get("sentiment", {})
                
                if sentiment:
                    col1, col2, col3 = st.columns(3)
                    
                    counts = sentiment.get("counts", {})
                    percentages = sentiment.get("percentages", {})
                    
                    with col1:
                        st.metric(
                            "🟢 Positive",
                            counts.get("positive", 0),
                            delta=f"{percentages.get('positive', 0):.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "🟡 Neutral",
                            counts.get("neutral", 0),
                            delta=f"{percentages.get('neutral', 0):.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "🔴 Negative",
                            counts.get("negative", 0),
                            delta=f"{percentages.get('negative', 0):.1f}%"
                        )
                    
                    # Sentiment index
                    sentiment_index = sentiment.get("sentiment_index", 0)
                    ci = sentiment.get("confidence_interval_95", (0, 0))
                    
                    st.metric(
                        "Sentiment Index",
                        f"{sentiment_index:.3f}",
                        help=f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
                    )
                    
                    # Interpretation
                    if sentiment_index > 0.3:
                        st.success("**Strong positive** sentiment across news and filings.")
                    elif sentiment_index > 0:
                        st.info("**Moderately positive** sentiment.")
                    elif sentiment_index > -0.3:
                        st.warning("**Moderately negative** sentiment.")
                    else:
                        st.error("**Strong negative** sentiment.")
                else:
                    st.info("No sentiment data available")
            
            with tab5:
                st.subheader("Data Sources")
                
                # Collect all citations
                summaries = result.get("summaries", {})
                all_citations = []
                
                for summary_result in summaries.values():
                    citations = summary_result.get("citations", [])
                    for citation in citations:
                        if citation not in all_citations:
                            all_citations.append(citation)
                
                if all_citations:
                    for i, citation in enumerate(all_citations, 1):
                        source = citation.get("source", "Unknown")
                        url = citation.get("url", "#")
                        date = citation.get("date", "N/A")
                        st.markdown(f"{i}. [{source} - {date}]({url})")
                else:
                    st.info("No sources available")
    
    else:
        # Show instructions when not generating
        st.info("""
        ### 🚀 Getting Started
        
        1. Enter a stock ticker in the sidebar (e.g., AAPL, MSFT, TSLA, NVDA)
        2. Configure options (cache, lookback period, format)
        3. Click **"Generate Brief"** to create your analyst report
        
        The system will:
        - Fetch recent SEC filings (10-K, 10-Q, 8-K)
        - Collect and analyze financial news
        - Generate grounded summaries with citations
        - Perform sentiment analysis using FinBERT
        - Create a comprehensive 2-page analyst brief
        
        **Note:** First-time generation for a ticker may take 30-60 seconds. 
        Subsequent runs with cache enabled will be much faster.
        """)
        
        # System requirements
        with st.expander("⚙️ System Requirements"):
            st.markdown("""
            **Required:**
            - Python 3.12+
            - Azure OpenAI API access
            - Internet connection for SEC EDGAR and news
            
            **Models:**
            - LLM: Azure OpenAI GPT-4o-mini
            - Embeddings: Azure OpenAI text-embedding-ada-002
            - Sentiment: FinBERT (`ProsusAI/finbert`)
            
            **Configuration:**
            - Set Azure OpenAI credentials in `.env` file
            - See `QUICKSTART_AZURE.md` for setup guide
            """)


if __name__ == "__main__":
    main()
