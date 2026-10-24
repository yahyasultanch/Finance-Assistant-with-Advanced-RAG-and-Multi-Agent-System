"""Streamlit UI for AI Finance Assistant."""
import streamlit as st
import json
import sys
import uuid
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
    AZURE_OPENAI_API_VERSION, FINBERT_MODEL, RERANKER_MODEL,
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
        reranker_model=RERANKER_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K_RETRIEVAL
    )


def render_review(review):
    """Human-in-the-loop panel. Returns (approved, note) once a button is pressed, else None."""
    st.subheader("🧑‍⚖️ Review the draft before publishing")
    target = review.get("price_target")
    target = f"${target:.2f}" if isinstance(target, (int, float)) else "N/A"
    st.markdown(
        f"**Draft rating:** {review['rating']} (score {review['score']:.1f}, 12M target {target}) · "
        f"**Citation coverage:** {review['coverage']:.0%}"
    )
    st.dataframe(review["claims"], use_container_width=True, hide_index=True)
    note = st.text_area("Reviewer note (added to the report)", key="review_note")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve and publish", type="primary", use_container_width=True):
        return True, note
    if col2.button("❌ Reject", use_container_width=True):
        return False, note
    return None


def render_trace(result):
    """What the agents did: plan, verification, tool calls, token usage."""
    st.markdown(
        "**Graph:** ingest → planner → researchers *(parallel, tool-calling)* → analyst → "
        "verifier ⟲ *(replans if coverage < 80%)* → score → review → report"
    )
    verification = result.get("verification", {})
    usage = result.get("usage", {})
    if verification:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Claims verified", f"{verification['verified']}/{verification['total']}")
        col2.metric("Citation coverage", f"{verification['coverage']:.0%}",
                    f"{verification['coverage'] - verification['first_pass_coverage']:+.0%} vs first draft")
        col3.metric("Revision rounds", verification.get("revisions", 0))
        tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
        col4.metric("LLM calls / tokens", f"{usage.get('calls', 0)} / {tokens / 1000:.1f}k")

    st.subheader("Research plan")
    st.dataframe(result.get("plan", []), use_container_width=True, hide_index=True)

    st.subheader("Claim verification")
    st.dataframe(
        [{"section": c["section"], "status": c["status"], "claim": c["text"],
          "sources": ", ".join(c["sources"]), "reason": c.get("reason", "")}
         for c in verification.get("claims", [])],
        use_container_width=True, hide_index=True
    )

    st.subheader("Tool calls")
    st.dataframe(
        [{"node": e["node"], "tool": e.get("tool", ""), "args": json.dumps(e.get("args", "")),
          "sources": str(e.get("sources", "")), "ms": str(e.get("ms", "")),
          "error": e.get("error") or "", "detail": e.get("detail", "")}
         for e in result.get("trace", [])],
        use_container_width=True, hide_index=True
    )


def render_chat(ticker):
    """Follow-up Q&A; history lives in the LangGraph checkpointer under this thread id."""
    orchestrator = get_orchestrator()
    thread_key = f"chat_thread_{ticker}"
    if thread_key not in st.session_state:
        st.session_state[thread_key] = f"chat-{ticker}-{uuid.uuid4().hex[:8]}"
    thread_id = st.session_state[thread_key]

    st.caption("Ask follow-up questions. The assistant calls the same research tools and "
               "remembers this conversation.")
    for message in orchestrator.chat_history(thread_id):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(" · ".join(f"[{s['id']}: {s['label']}]({s['url']})" for s in message["sources"]))
            if message.get("tool_calls"):
                st.caption("Tools: " + ", ".join(c["tool"] for c in message["tool_calls"]))

    question = st.chat_input(f"Ask about {ticker}...", key=f"chat_input_{ticker}")
    if question:
        with st.spinner("Researching..."):
            orchestrator.chat(ticker, question, thread_id)
        st.rerun()


def render_track_record(ticker):
    """Long-term memory: every published brief and how its price calls turned out."""
    record = get_orchestrator().track_record(ticker)
    summary = record["summary"]

    st.caption("Every published brief is remembered, and its price calls are scored against "
               "actual closes once their horizon passes.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Calls logged", summary["total"], f"{summary['pending']} pending", delta_color="off")
    col2.metric("Direction correct",
                f"{summary['direction_hit_rate']:.0%}" if summary.get("scored") else "n/a")
    col3.metric("Mean abs. error",
                f"{summary['mean_abs_error_pct']:.1f}%" if summary.get("scored") else "n/a")

    st.subheader("Prediction ledger")
    names = {"7d": "7-day", "12m": "12-month", "rating_target": "Rating target"}
    st.dataframe(
        [{"made on": e["made_on"], "call": names.get(e["kind"], e["kind"]),
          "price then": e["price"], "target": e["target"], "due": e["due"],
          "status": e["status"], "actual / latest": e["actual"],
          "error %": e["error_pct"], "direction correct": e["direction_hit"]}
         for e in reversed(record["entries"])],
        use_container_width=True, hide_index=True
    )

    st.subheader("Earlier briefs (memory)")
    st.dataframe(
        [{"date": b["date"], "rating": b["rating"], "score": b["score"],
          "price target": b["price_target"], "price then": b["price"],
          "citation coverage": f"{b['coverage']:.0%}"}
         for b in record["briefs"]],
        use_container_width=True, hide_index=True
    )


def main():
    """Main Streamlit app."""
    
    # Title and description
    st.title("📊 AI Investment Research Assistant")
    st.markdown("""
    Generate comprehensive analyst briefs using RAG, AI agents, and financial sentiment analysis.
    
    **Features:**
    - 🧭 LangGraph agents: planner, parallel tool-calling researchers, analyst, verifier
    - 📑 SEC filing analysis (10-K, 10-Q, 8-K press releases, XBRL financials)
    - 📰 News aggregation and sentiment
    - ✅ Every claim's citations verified before it reaches the report
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

        review = st.checkbox(
            "Review before publishing",
            value=False,
            help="Human-in-the-loop: the agent graph pauses after verification and scoring "
                 "so you can approve or reject the draft before the report is written"
        )

        st.divider()
        
        # Generate brief button
        generate_button = st.button("🚀 Generate Brief", type="primary", use_container_width=True)
        
        if generate_button:
            # Store input params in session state
            st.session_state['ticker'] = ticker
            st.session_state['use_cache'] = use_cache
            st.session_state['days_back'] = days_back
            st.session_state['save_format'] = save_format
            st.session_state['review'] = review
            st.session_state['generate_clicked'] = True
            st.session_state['result'] = None  # Clear previous result
            st.session_state['pending_review'] = None
        
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
                pending = st.session_state.get('pending_review')

                if pending is None:
                    with st.spinner("Agents at work: ingesting filings, planning, researching in "
                                    "parallel, drafting and verifying citations..."):
                        outcome = orchestrator.generate_brief(
                            ticker=ticker,
                            use_cache=use_cache,
                            days_back=days_back,
                            save_format=save_format,
                            review=st.session_state.get('review', False)
                        )
                    if outcome["status"] == "needs_review":
                        st.session_state['pending_review'] = pending = outcome
                    else:
                        st.session_state['result'] = outcome

                # Graph is paused at its review node; resume with the decision
                if pending is not None:
                    decision = render_review(pending["review"])
                    if decision is None:
                        return
                    approved, note = decision
                    with st.spinner("Resuming the agent graph..."):
                        outcome = orchestrator.resume_brief(pending["thread_id"], approved, note)
                    st.session_state['pending_review'] = None
                    if outcome["status"] != "done":
                        st.warning("Draft rejected: no report was written.")
                        st.session_state['generate_clicked'] = False
                        return
                    st.session_state['result'] = outcome
                    st.rerun()  # redraw without the review panel
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
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📝 Report", "📈 Charts", "📊 Metrics", "💬 Sentiment", "🔍 Sources",
                "🧭 Agent Trace", "🗨️ Ask", "📒 Track Record"
            ])
            
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
                    tiers = {1: "SEC", 2: "Market data", 3: "News", 4: "Social"}
                    for i, citation in enumerate(all_citations, 1):
                        number = citation.get("number", i)
                        source = citation.get("source", "Unknown")
                        url = citation.get("url", "#")
                        date = citation.get("date", "N/A")
                        tier = tiers.get(citation.get("tier"), "")
                        st.markdown(f"{number}. [{source} - {date}]({url})" + (f" · *{tier}*" if tier else ""))
                else:
                    st.info("No sources available")

            with tab6:
                render_trace(result)

            with tab7:
                render_chat(ticker)

            with tab8:
                render_track_record(ticker)

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

if __name__ == "__main__":
    main()
