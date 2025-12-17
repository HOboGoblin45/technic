"""
Cache Metrics Component
Real-time cache performance visualization and monitoring
Path 1 Task 2: Cache Status Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class CacheMetrics:
    """
    Cache performance metrics visualization component
    
    Features:
    - Real-time cache hit rate display
    - Cache size and memory usage
    - TTL settings visualization
    - Performance trends over time
    - Connection pool status
    - Response time comparisons
    """
    
    def __init__(self, api_url: str = "http://localhost:8003"):
        """
        Initialize cache metrics component
        
        Args:
            api_url: Base URL for the monitoring API
        """
        self.api_url = api_url
        self.cache_data = None
        self.connection_data = None
        self.performance_data = None
    
    def fetch_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch cache statistics from API"""
        try:
            response = requests.get(f"{self.api_url}/performance/cache", timeout=5)
            response.raise_for_status()
            self.cache_data = response.json()
            return self.cache_data
        except Exception as e:
            st.error(f"Error fetching cache stats: {e}")
            return None
    
    def fetch_connection_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch connection pool statistics from API"""
        try:
            response = requests.get(f"{self.api_url}/performance/connections", timeout=5)
            response.raise_for_status()
            self.connection_data = response.json()
            return self.connection_data
        except Exception as e:
            st.error(f"Error fetching connection stats: {e}")
            return None
    
    def fetch_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Fetch overall performance summary from API"""
        try:
            response = requests.get(f"{self.api_url}/performance/summary", timeout=5)
            response.raise_for_status()
            self.performance_data = response.json()
            return self.performance_data
        except Exception as e:
            st.error(f"Error fetching performance summary: {e}")
            return None
    
    def render_overview(self):
        """Render cache overview metrics"""
        st.header("📊 Cache Performance Overview")
        
        # Fetch data
        cache_stats = self.fetch_cache_stats()
        perf_stats = self.fetch_performance_summary()
        
        if not cache_stats:
            st.warning("Cache statistics not available")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hit_rate = cache_stats.get('hit_rate', 0)
            target = 70.0
            delta = hit_rate - target
            
            st.metric(
                "Cache Hit Rate",
                f"{hit_rate:.1f}%",
                delta=f"{delta:+.1f}% vs target",
                delta_color="normal" if hit_rate >= target else "inverse"
            )
        
        with col2:
            total_requests = cache_stats.get('total_requests', 0)
            st.metric(
                "Total Requests",
                f"{total_requests:,}",
                delta=f"{cache_stats.get('cache_hits', 0):,} hits"
            )
        
        with col3:
            cache_size = cache_stats.get('cache_size', 0)
            st.metric(
                "Cached Items",
                f"{cache_size:,}",
                delta="Active keys"
            )
        
        with col4:
            if perf_stats:
                speedup = perf_stats.get('response_times', {}).get('speedup', '1.0x')
                st.metric(
                    "Performance Gain",
                    speedup,
                    delta="vs uncached"
                )
    
    def render_hit_rate_gauge(self):
        """Render cache hit rate as a gauge chart"""
        if not self.cache_data:
            return
        
        hit_rate = self.cache_data.get('hit_rate', 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=hit_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cache Hit Rate", 'font': {'size': 24}},
            delta={'reference': 70, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100], 'ticksuffix': '%'},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_cache_breakdown(self):
        """Render cache hits vs misses breakdown"""
        if not self.cache_data:
            return
        
        hits = self.cache_data.get('cache_hits', 0)
        misses = self.cache_data.get('cache_misses', 0)
        
        if hits == 0 and misses == 0:
            st.info("No cache activity yet")
            return
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Cache Hits', 'Cache Misses'],
            values=[hits, misses],
            hole=.4,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        
        fig.update_layout(
            title="Cache Hits vs Misses",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ttl_settings(self):
        """Render TTL settings for different cache types"""
        if not self.cache_data:
            return
        
        st.subheader("⏱️ Cache TTL Settings")
        
        ttl_settings = self.cache_data.get('ttl_seconds', {})
        
        if not ttl_settings:
            st.info("No TTL settings available")
            return
        
        # Create DataFrame
        ttl_df = pd.DataFrame([
            {'Endpoint': key.replace('_', ' ').title(), 'TTL (seconds)': value}
            for key, value in ttl_settings.items()
        ])
        
        # Bar chart
        fig = px.bar(
            ttl_df,
            x='Endpoint',
            y='TTL (seconds)',
            title='Cache TTL by Endpoint',
            color='TTL (seconds)',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table view
        with st.expander("📋 Detailed TTL Settings"):
            st.dataframe(ttl_df, use_container_width=True, hide_index=True)
    
    def render_connection_pool(self):
        """Render connection pool status"""
        st.subheader("🔌 Connection Pool Status")
        
        conn_stats = self.fetch_connection_stats()
        
        if not conn_stats:
            st.warning("Connection pool statistics not available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            active = conn_stats.get('active_connections', 0)
            max_conn = conn_stats.get('max_connections', 20)
            st.metric(
                "Active Connections",
                f"{active}/{max_conn}",
                delta=f"{(active/max_conn*100):.0f}% utilized"
            )
        
        with col2:
            idle = conn_stats.get('idle_connections', 0)
            st.metric(
                "Idle Connections",
                f"{idle}",
                delta="Available"
            )
        
        with col3:
            reuse_rate = conn_stats.get('connection_reuse_rate', 0)
            st.metric(
                "Reuse Rate",
                f"{reuse_rate:.1f}%",
                delta="Efficiency"
            )
        
        # Connection pool visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Active',
            x=['Connections'],
            y=[active],
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='Idle',
            x=['Connections'],
            y=[idle],
            marker_color='#95a5a6'
        ))
        
        fig.update_layout(
            title='Connection Pool Distribution',
            barmode='stack',
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title='Count',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_response_times(self):
        """Render response time comparison"""
        st.subheader("⚡ Response Time Analysis")
        
        if not self.performance_data:
            self.fetch_performance_summary()
        
        if not self.performance_data:
            st.warning("Performance data not available")
            return
        
        response_times = self.performance_data.get('response_times', {})
        
        avg_cached = response_times.get('avg_cached', 0)
        avg_uncached = response_times.get('avg_uncached', 0)
        speedup = response_times.get('speedup', '1.0x')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Cached Response",
                f"{avg_cached:.0f}ms",
                delta="Fast"
            )
        
        with col2:
            st.metric(
                "Uncached Response",
                f"{avg_uncached:.0f}ms",
                delta="Baseline"
            )
        
        with col3:
            st.metric(
                "Speedup",
                speedup,
                delta="Performance gain"
            )
        
        # Comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Response Time (ms)',
            x=['Cached', 'Uncached'],
            y=[avg_cached, avg_uncached],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f"{avg_cached:.0f}ms", f"{avg_uncached:.0f}ms"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Cached vs Uncached Response Times',
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title='Response Time (ms)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_summary(self):
        """Render overall performance summary"""
        st.subheader("📈 Performance Summary")
        
        if not self.performance_data:
            self.fetch_performance_summary()
        
        if not self.performance_data:
            st.warning("Performance summary not available")
            return
        
        cache_info = self.performance_data.get('cache', {})
        conn_info = self.performance_data.get('connections', {})
        
        # Summary table
        summary_data = {
            'Metric': [
                'Cache Hit Rate',
                'Total Requests',
                'Connection Reuse',
                'Active Connections',
                'Performance Gain'
            ],
            'Value': [
                f"{cache_info.get('hit_rate', 0):.1f}%",
                f"{cache_info.get('total_requests', 0):,}",
                f"{conn_info.get('reuse_rate', 0):.1f}%",
                f"{conn_info.get('active', 0)}",
                self.performance_data.get('response_times', {}).get('speedup', 'N/A')
            ],
            'Status': [
                '✅ Good' if cache_info.get('hit_rate', 0) >= 70 else '⚠️ Low',
                '✅ Active',
                '✅ Efficient' if conn_info.get('reuse_rate', 0) >= 90 else '⚠️ Low',
                '✅ Normal',
                '✅ Optimized'
            ]
        }
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_recommendations(self):
        """Render optimization recommendations"""
        st.subheader("💡 Optimization Recommendations")
        
        if not self.cache_data:
            return
        
        hit_rate = self.cache_data.get('hit_rate', 0)
        cache_size = self.cache_data.get('cache_size', 0)
        
        recommendations = []
        
        # Hit rate recommendations
        if hit_rate < 50:
            recommendations.append({
                'priority': '🔴 High',
                'issue': 'Low cache hit rate',
                'recommendation': 'Increase TTL values or implement cache warming',
                'impact': 'High performance improvement'
            })
        elif hit_rate < 70:
            recommendations.append({
                'priority': '🟡 Medium',
                'issue': 'Below target hit rate',
                'recommendation': 'Review cache invalidation strategy',
                'impact': 'Moderate performance improvement'
            })
        else:
            recommendations.append({
                'priority': '🟢 Good',
                'issue': 'Cache performing well',
                'recommendation': 'Monitor and maintain current settings',
                'impact': 'Maintain performance'
            })
        
        # Cache size recommendations
        if cache_size > 1000:
            recommendations.append({
                'priority': '🟡 Medium',
                'issue': 'Large cache size',
                'recommendation': 'Consider implementing cache eviction policy',
                'impact': 'Reduce memory usage'
            })
        
        # Connection pool recommendations
        if self.connection_data:
            reuse_rate = self.connection_data.get('connection_reuse_rate', 0)
            if reuse_rate < 80:
                recommendations.append({
                    'priority': '🟡 Medium',
                    'issue': 'Low connection reuse',
                    'recommendation': 'Increase connection pool size or timeout',
                    'impact': 'Reduce connection overhead'
                })
        
        # Display recommendations
        for rec in recommendations:
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**{rec['priority']}**")
                with col2:
                    st.markdown(f"**{rec['issue']}**")
                    st.markdown(f"→ {rec['recommendation']}")
                    st.caption(f"Impact: {rec['impact']}")
                st.markdown("---")
    
    def render_full_dashboard(self):
        """Render complete cache metrics dashboard"""
        st.title("🗄️ Cache Performance Dashboard")
        st.markdown("Real-time cache and connection pool monitoring")
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        st.markdown("---")
        
        # Overview
        self.render_overview()
        
        st.markdown("---")
        
        # Main metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_hit_rate_gauge()
        
        with col2:
            self.render_cache_breakdown()
        
        st.markdown("---")
        
        # TTL and Connection Pool
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_ttl_settings()
        
        with col2:
            self.render_connection_pool()
        
        st.markdown("---")
        
        # Response times
        self.render_response_times()
        
        st.markdown("---")
        
        # Performance summary
        self.render_performance_summary()
        
        st.markdown("---")
        
        # Recommendations
        self.render_recommendations()
        
        # Auto-refresh
        if auto_refresh:
            import time
            time.sleep(5)
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cache Metrics Dashboard",
        page_icon="🗄️",
        layout="wide"
    )
    
    metrics = CacheMetrics()
    metrics.render_full_dashboard()
