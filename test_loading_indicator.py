"""
Test Loading Indicator Component
Demonstrates the loading indicator in action
Path 1 Task 1: Loading Indicators - Testing
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Add components directory to path
sys.path.insert(0, str(Path(__file__).parent))

from components.LoadingIndicator import LoadingIndicator

st.set_page_config(
    page_title="Loading Indicator Test",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Loading Indicator Component Test")
st.markdown("Test the loading indicator with different scenarios")

# Create tabs for different test scenarios
tab1, tab2, tab3, tab4 = st.tabs([
    "Basic Progress",
    "Multi-Stage Scan",
    "Simple Loading",
    "Error Handling"
])

# Tab 1: Basic Progress
with tab1:
    st.header("Basic Progress Indicator")
    st.markdown("Shows a simple progress bar with percentage")
    
    if st.button("Start Basic Test", key="basic"):
        indicator = LoadingIndicator()
        indicator.start(total_items=100)
        
        progress_container = st.empty()
        
        with progress_container.container():
            for i in range(0, 101, 5):
                indicator.update({
                    'stage': 'symbol_scanning',
                    'current': i,
                    'total': 100,
                    'message': f'Processing item {i}/100...',
                    'metadata': {
                        'overall_progress_pct': i,
                        'stage_progress_pct': i,
                        'overall_eta': (100 - i) * 0.1,
                        'cache_hit_rate': 75.5,
                        'stage_throughput': 10.0
                    }
                })
                indicator.render()
                time.sleep(0.2)
        
        progress_container.empty()
        indicator.complete("Basic test complete!")

# Tab 2: Multi-Stage Scan
with tab2:
    st.header("Multi-Stage Scanner Simulation")
    st.markdown("Simulates a full scanner run with all stages")
    
    if st.button("Start Scanner Simulation", key="scanner"):
        indicator = LoadingIndicator()
        indicator.start(total_items=100)
        
        progress_container = st.empty()
        
        # Define stages with their progress ranges
        stages = [
            {
                'name': 'initializing',
                'start': 0,
                'end': 5,
                'message': 'Initializing scanner...'
            },
            {
                'name': 'universe_loading',
                'start': 5,
                'end': 10,
                'message': 'Loading universe symbols...'
            },
            {
                'name': 'data_fetching',
                'start': 10,
                'end': 30,
                'message': 'Fetching price data...'
            },
            {
                'name': 'symbol_scanning',
                'start': 30,
                'end': 95,
                'message': 'Scanning symbols...'
            },
            {
                'name': 'finalization',
                'start': 95,
                'end': 100,
                'message': 'Finalizing results...'
            }
        ]
        
        for stage in stages:
            for progress in range(stage['start'], stage['end'] + 1, 1):
                with progress_container.container():
                    # Calculate stage-specific progress
                    stage_range = stage['end'] - stage['start']
                    stage_progress = ((progress - stage['start']) / stage_range * 100) if stage_range > 0 else 100
                    
                    # Simulate varying cache hit rates
                    cache_hit_rate = 50 + (progress * 0.3)  # Increases over time
                    
                    # Simulate throughput
                    throughput = 2.5 if stage['name'] == 'symbol_scanning' else 10.0
                    
                    indicator.update({
                        'stage': stage['name'],
                        'current': progress,
                        'total': 100,
                        'message': f"{stage['message']} ({progress}%)",
                        'metadata': {
                            'overall_progress_pct': progress,
                            'stage_progress_pct': stage_progress,
                            'overall_eta': (100 - progress) * 0.5,
                            'cache_hit_rate': min(cache_hit_rate, 95),
                            'stage_throughput': throughput
                        }
                    })
                    indicator.render()
                
                time.sleep(0.1)
        
        progress_container.empty()
        indicator.complete(
            "Scanner complete! Found 42 high-quality opportunities.",
            show_summary=True
        )

# Tab 3: Simple Loading
with tab3:
    st.header("Simple Loading Indicator")
    st.markdown("Minimal loading indicator without detailed progress")
    
    if st.button("Start Simple Loading", key="simple"):
        indicator = LoadingIndicator()
        
        progress_container = st.empty()
        
        with progress_container.container():
            for i in range(0, 101, 10):
                indicator.render_simple(
                    message="Processing data...",
                    progress=i
                )
                time.sleep(0.3)
        
        progress_container.empty()
        st.success("✅ Processing complete!")

# Tab 4: Error Handling
with tab4:
    st.header("Error Handling")
    st.markdown("Demonstrates error states and recovery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Simulate Error", key="error"):
            indicator = LoadingIndicator()
            indicator.start(total_items=100)
            
            progress_container = st.empty()
            
            # Progress normally for a bit
            with progress_container.container():
                for i in range(0, 51, 10):
                    indicator.update({
                        'stage': 'symbol_scanning',
                        'current': i,
                        'total': 100,
                        'message': f'Processing... {i}%',
                        'metadata': {
                            'overall_progress_pct': i,
                            'stage_progress_pct': i * 2
                        }
                    })
                    indicator.render()
                    time.sleep(0.3)
            
            # Then show error
            progress_container.empty()
            indicator.error(
                "Connection timeout while fetching data",
                details="Error: requests.exceptions.Timeout: HTTPConnectionPool(host='api.example.com', port=443): Read timed out. (read timeout=30)"
            )
    
    with col2:
        if st.button("Simulate Recovery", key="recovery"):
            indicator = LoadingIndicator()
            indicator.start(total_items=100)
            
            progress_container = st.empty()
            
            # Progress normally
            with progress_container.container():
                for i in range(0, 51, 10):
                    indicator.update({
                        'stage': 'data_fetching',
                        'current': i,
                        'total': 100,
                        'message': f'Fetching data... {i}%',
                        'metadata': {
                            'overall_progress_pct': i,
                            'stage_progress_pct': i * 2
                        }
                    })
                    indicator.render()
                    time.sleep(0.2)
            
            # Show warning
            progress_container.empty()
            st.warning("⚠️ Connection slow - retrying...")
            time.sleep(1)
            
            # Continue with progress
            progress_container2 = st.empty()
            with progress_container2.container():
                for i in range(51, 101, 10):
                    indicator.update({
                        'stage': 'data_fetching',
                        'current': i,
                        'total': 100,
                        'message': f'Retrying... {i}%',
                        'metadata': {
                            'overall_progress_pct': i,
                            'stage_progress_pct': i * 2
                        }
                    })
                    indicator.render()
                    time.sleep(0.2)
            
            progress_container2.empty()
            indicator.complete("Recovered successfully!")

# Additional information
st.markdown("---")
st.header("📚 Component Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Visual Features:**
    - Progress bars
    - Stage indicators
    - Icon-based stages
    - Color-coded status
    """)

with col2:
    st.markdown("""
    **Information Display:**
    - Current stage
    - Progress percentage
    - ETA calculation
    - Processing speed
    """)

with col3:
    st.markdown("""
    **Advanced Features:**
    - Multi-stage tracking
    - Cache statistics
    - Error handling
    - Expandable details
    """)

st.markdown("---")
st.markdown("**Usage:** Import `LoadingIndicator` from `components/LoadingIndicator.py` and integrate into your dashboard.")
