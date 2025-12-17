"""
Standalone Cache Performance Dashboard
Real-time monitoring of cache and connection pool performance
Path 1 Task 2: Cache Status Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add components directory to path
sys.path.insert(0, str(Path(__file__).parent))

from components.CacheMetrics import CacheMetrics

# Page configuration
st.set_page_config(
    page_title="Cache Performance Dashboard",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-good {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-warning {
        color: #f39c12;
        font-weight: bold;
    }
    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }
    .recommendation-high {
        background-color: #ffebee;
        padding: 10px;
        border-left: 4px solid #f44336;
        margin: 5px 0;
    }
    .recommendation-medium {
        background-color: #fff3e0;
        padding: 10px;
        border-left: 4px solid #ff9800;
        margin: 5px 0;
    }
    .recommendation-good {
        background-color: #e8f5e9;
        padding: 10px;
        border-left: 4px solid #4caf50;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    
    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8003",
        help="Base URL for the monitoring API"
    )
    
    st.markdown("---")
    
    # Refresh settings
    st.header("🔄 Refresh Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=1,
            max_value=30,
            value=5
        )
    
    st.markdown("---")
    
    # Display options
    st.header("📊 Display Options")
    show_overview = st.checkbox("Overview Metrics", value=True)
    show_gauges = st.checkbox("Gauge Charts", value=True)
    show_ttl = st.checkbox("TTL Settings", value=True)
    show_connections = st.checkbox("Connection Pool", value=True)
    show_response_times = st.checkbox("Response Times", value=True)
    show_summary = st.checkbox("Performance Summary", value=True)
    show_recommendations = st.checkbox("Recommendations", value=True)
    
    st.markdown("---")
    
    # Quick actions
    st.header("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    
    if st.button("📊 View API Docs", use_container_width=True):
        st.markdown(f"[Open API Documentation]({api_url}/docs)")
    
    st.markdown("---")
    
    # Info
    st.header("ℹ️ Information")
    st.markdown("""
    **Cache Dashboard**
    
    Monitor cache performance in real-time:
    - Hit rates and efficiency
    - TTL configurations
    - Connection pool status
    - Response time analysis
    - Optimization recommendations
    
    **Target Metrics:**
    - Hit Rate: >70%
    - Connection Reuse: >90%
    - Response Time: <100ms
    """)

# Main content
def main():
    # Initialize cache metrics component
    metrics = CacheMetrics(api_url=api_url)
    
    # Title
    st.title("🗄️ Cache Performance Dashboard")
    st.markdown("Real-time monitoring of cache and connection pool performance")
    
    # Refresh button in main area
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        if st.button("🔄 Refresh", key="main_refresh"):
            st.rerun()
    with col2:
        st.markdown(f"**API:** {api_url}")
    
    st.markdown("---")
    
    # Render selected sections
    if show_overview:
        metrics.render_overview()
        st.markdown("---")
    
    if show_gauges:
        col1, col2 = st.columns(2)
        with col1:
            metrics.render_hit_rate_gauge()
        with col2:
            metrics.render_cache_breakdown()
        st.markdown("---")
    
    if show_ttl or show_connections:
        col1, col2 = st.columns(2)
        with col1:
            if show_ttl:
                metrics.render_ttl_settings()
        with col2:
            if show_connections:
                metrics.render_connection_pool()
        st.markdown("---")
    
    if show_response_times:
        metrics.render_response_times()
        st.markdown("---")
    
    if show_summary:
        metrics.render_performance_summary()
        st.markdown("---")
    
    if show_recommendations:
        metrics.render_recommendations()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Last Updated:**")
        from datetime import datetime
        st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with col2:
        st.markdown("**Dashboard Version:**")
        st.markdown("1.0.0")
    
    with col3:
        st.markdown("**Status:**")
        # Check if API is accessible
        try:
            import requests
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                st.markdown('<span class="status-good">● Online</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error">● Issues</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span class="status-error">● Offline</span>', unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
