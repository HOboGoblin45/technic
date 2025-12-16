"""
Real-Time Monitoring Dashboard
Streamlit-based UI for ML API monitoring
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8003"

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-error {
        background-color: #ffebee;
        padding: 10px;
        border-left: 4px solid #f44336;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 10px;
        border-left: 4px solid #ff9800;
        margin: 5px 0;
    }
    .alert-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-left: 4px solid #2196f3;
        margin: 5px 0;
    }
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def fetch_data(endpoint):
    """Fetch data from monitoring API"""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from {endpoint}: {e}")
        return None


def format_uptime(seconds):
    """Format uptime in human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def create_gauge_chart(value, title, max_value=100, color_threshold=None):
    """Create a gauge chart"""
    if color_threshold is None:
        color_threshold = [0, 50, 75, 100]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, color_threshold[1]], 'color': "lightgreen"},
                {'range': [color_threshold[1], color_threshold[2]], 'color': "yellow"},
                {'range': [color_threshold[2], max_value], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': color_threshold[2]
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    """Main dashboard function"""
    
    # Header
    st.title("🔍 ML API Monitoring Dashboard")
    st.markdown("Real-time monitoring for ML-powered scan optimization")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, 5)
        
        st.markdown("---")
        st.header("📡 API Status")
        
        # Check API health
        health_data = fetch_data("/health")
        if health_data:
            status = health_data.get('status', 'unknown')
            if status == 'healthy':
                st.markdown('<p class="status-healthy">● HEALTHY</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">● UNHEALTHY</p>', unsafe_allow_html=True)
            
            uptime = health_data.get('uptime_seconds', 0)
            st.metric("Uptime", format_uptime(uptime))
        else:
            st.markdown('<p class="status-error">● OFFLINE</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Endpoints:**")
        st.markdown("- Monitoring API: :8003")
        st.markdown("- ML API: :8002")
        st.markdown("- Documentation: [/docs](http://localhost:8003/docs)")
    
    # Main content
    # Fetch current metrics
    metrics_data = fetch_data("/metrics/current")
    alert_data = fetch_data("/alerts/active")
    summary_data = fetch_data("/metrics/summary")
    
    if not metrics_data:
        st.error("Unable to fetch metrics. Please ensure the Monitoring API is running.")
        return
    
    metrics = metrics_data.get('metrics', {})
    api_metrics = metrics.get('api_metrics', {})
    model_metrics = metrics.get('model_metrics', {})
    system_metrics = metrics.get('system_metrics', {})
    
    # Overview Section
    st.header("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{api_metrics.get('total_requests', 0):,}",
            delta=f"{api_metrics.get('requests_per_minute', 0):.1f} req/min"
        )
    
    with col2:
        response_time = api_metrics.get('avg_response_time_ms', 0)
        st.metric(
            "Avg Response Time",
            f"{response_time:.0f}ms",
            delta=f"P95: {api_metrics.get('p95_response_time_ms', 0):.0f}ms"
        )
    
    with col3:
        error_rate = api_metrics.get('error_rate_percent', 0)
        st.metric(
            "Error Rate",
            f"{error_rate:.2f}%",
            delta=f"{api_metrics.get('total_errors', 0)} errors",
            delta_color="inverse"
        )
    
    with col4:
        active_alerts = len(alert_data.get('alerts', [])) if alert_data else 0
        st.metric(
            "Active Alerts",
            active_alerts,
            delta="Critical" if active_alerts > 0 else "All clear",
            delta_color="inverse" if active_alerts > 0 else "normal"
        )
    
    # Performance Gauges
    st.header("⚡ Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_gauge_chart(
            response_time,
            "Response Time (ms)",
            max_value=1000,
            color_threshold=[0, 200, 500, 1000]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(
            error_rate,
            "Error Rate (%)",
            max_value=10,
            color_threshold=[0, 1, 5, 10]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        cpu_percent = system_metrics.get('cpu_percent', 0)
        fig = create_gauge_chart(
            cpu_percent,
            "CPU Usage (%)",
            max_value=100,
            color_threshold=[0, 50, 80, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    st.header("🤖 Model Performance")
    
    if model_metrics:
        cols = st.columns(len(model_metrics))
        
        for idx, (model_name, stats) in enumerate(model_metrics.items()):
            with cols[idx]:
                st.subheader(model_name.replace('_', ' ').title())
                
                predictions = stats.get('predictions_count', 0)
                mae = stats.get('avg_mae', 0)
                confidence = stats.get('avg_confidence', 0)
                
                st.metric("Predictions", f"{predictions:,}")
                st.metric("Avg MAE", f"{mae:.2f}")
                st.metric("Avg Confidence", f"{confidence:.1%}")
                
                # Status indicator
                if mae < 5:
                    st.success("✓ Excellent")
                elif mae < 10:
                    st.info("○ Good")
                else:
                    st.warning("⚠ Needs attention")
    else:
        st.info("No model predictions yet")
    
    # System Resources
    st.header("💻 System Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        memory_mb = system_metrics.get('memory_usage_mb', 0)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Memory bar
        memory_percent = min(memory_mb / 1000 * 100, 100)
        st.progress(memory_percent / 100)
    
    with col2:
        cpu = system_metrics.get('cpu_percent', 0)
        st.metric("CPU Usage", f"{cpu:.1f}%")
        
        # CPU bar
        st.progress(cpu / 100)
    
    with col3:
        disk = system_metrics.get('disk_usage_percent', 0)
        st.metric("Disk Usage", f"{disk:.1f}%")
        
        # Disk bar
        st.progress(disk / 100)
    
    # Active Alerts
    st.header("🚨 Active Alerts")
    
    if alert_data and alert_data.get('alerts'):
        for alert in alert_data['alerts']:
            severity = alert['severity']
            message = alert['message']
            age = alert.get('age_seconds', 0)
            
            if severity == 'error':
                st.error(f"❌ **ERROR**: {message} (Age: {format_uptime(age)})")
            elif severity == 'warning':
                st.warning(f"⚠️ **WARNING**: {message} (Age: {format_uptime(age)})")
            else:
                st.info(f"ℹ️ **INFO**: {message} (Age: {format_uptime(age)})")
    else:
        st.success("✓ No active alerts - All systems operational")
    
    # Endpoint Statistics
    if summary_data:
        st.header("📈 Endpoint Statistics")
        
        summary = summary_data.get('summary', {})
        api_summary = summary.get('api', {})
        
        # Create a simple table
        stats_df = pd.DataFrame({
            'Metric': [
                'Total Requests',
                'Requests/Minute',
                'Avg Response Time',
                'P95 Response Time',
                'Error Rate'
            ],
            'Value': [
                f"{api_summary.get('total_requests', 0):,}",
                f"{api_summary.get('requests_per_minute', 0):.1f}",
                f"{api_summary.get('avg_response_time_ms', 0):.1f}ms",
                f"{api_summary.get('p95_response_time_ms', 0):.1f}ms",
                f"{api_summary.get('error_rate_percent', 0):.2f}%"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Last Updated:**")
        st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with col2:
        st.markdown("**API Version:**")
        st.markdown("1.0.0")
    
    with col3:
        st.markdown("**Status:**")
        if health_data and health_data.get('status') == 'healthy':
            st.markdown('<span class="status-healthy">● Operational</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">● Issues Detected</span>', unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
