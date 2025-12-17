"""
Loading Indicator Component
Shows real-time progress during scans with stage information and ETA
Path 1 Task 1: Loading Indicators
"""

import streamlit as st
import time
from typing import Optional, Dict, Any


class LoadingIndicator:
    """
    Reusable loading indicator component with multi-stage progress tracking
    
    Features:
    - Stage-by-stage progress display
    - ETA calculation and countdown
    - Visual progress bar
    - Current status messages
    - Cancel button (optional)
    """
    
    # Stage definitions with weights (must sum to 100)
    STAGES = {
        'initializing': {'name': 'Initializing', 'weight': 5, 'icon': '🔄'},
        'universe_loading': {'name': 'Loading Universe', 'weight': 5, 'icon': '📊'},
        'data_fetching': {'name': 'Fetching Data', 'weight': 20, 'icon': '📡'},
        'symbol_scanning': {'name': 'Scanning Symbols', 'weight': 65, 'icon': '🔍'},
        'finalization': {'name': 'Finalizing Results', 'weight': 5, 'icon': '✅'}
    }
    
    def __init__(self, container=None):
        """
        Initialize loading indicator
        
        Args:
            container: Streamlit container to render in (optional)
        """
        self.container = container or st
        self.start_time = None
        self.current_stage = None
        self.progress_data = {}
    
    def start(self, total_items: int = 100):
        """
        Start the loading indicator
        
        Args:
            total_items: Total number of items to process
        """
        self.start_time = time.time()
        self.total_items = total_items
        self.current_stage = 'initializing'
        self.progress_data = {
            'stage': 'initializing',
            'current': 0,
            'total': total_items,
            'message': 'Starting scan...',
            'overall_progress': 0
        }
    
    def update(self, progress_data: Dict[str, Any]):
        """
        Update progress with new data
        
        Args:
            progress_data: Dictionary containing:
                - stage: Current stage name
                - current: Current progress count
                - total: Total items
                - message: Status message
                - metadata: Additional metadata (optional)
        """
        self.progress_data = progress_data
        self.current_stage = progress_data.get('stage', 'initializing')
    
    def render(self):
        """Render the loading indicator"""
        if not self.progress_data:
            return
        
        stage = self.progress_data.get('stage', 'initializing')
        current = self.progress_data.get('current', 0)
        total = self.progress_data.get('total', 100)
        message = self.progress_data.get('message', '')
        metadata = self.progress_data.get('metadata', {})
        
        # Calculate overall progress
        overall_progress = metadata.get('overall_progress_pct', 0)
        stage_progress = metadata.get('stage_progress_pct', 0)
        
        # Get stage info
        stage_info = self.STAGES.get(stage, {'name': stage.title(), 'icon': '⏳', 'weight': 0})
        
        # Create layout
        with self.container:
            # Header with icon and stage name
            st.markdown(f"### {stage_info['icon']} {stage_info['name']}")
            
            # Overall progress bar
            st.progress(overall_progress / 100 if overall_progress <= 100 else 1.0)
            
            # Progress text
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Status:** {message}")
            
            with col2:
                st.markdown(f"**Progress:** {current}/{total}")
            
            with col3:
                # Calculate ETA
                eta = metadata.get('overall_eta')
                if eta:
                    eta_text = self._format_time(eta)
                    st.markdown(f"**ETA:** {eta_text}")
                else:
                    st.markdown(f"**ETA:** Calculating...")
            
            # Stage progress details
            with st.expander("📊 Stage Details", expanded=False):
                # Show all stages with checkmarks
                for stage_key, stage_data in self.STAGES.items():
                    if stage_key == stage:
                        # Current stage
                        st.markdown(
                            f"🔵 **{stage_data['icon']} {stage_data['name']}** "
                            f"({stage_progress:.1f}%)"
                        )
                    elif self._is_stage_complete(stage_key, stage):
                        # Completed stage
                        st.markdown(
                            f"✅ {stage_data['icon']} {stage_data['name']} "
                            f"(Complete)"
                        )
                    else:
                        # Pending stage
                        st.markdown(
                            f"⚪ {stage_data['icon']} {stage_data['name']} "
                            f"(Pending)"
                        )
            
            # Additional metadata
            if metadata:
                with st.expander("🔍 Technical Details", expanded=False):
                    # Cache stats
                    if 'cache_hit_rate' in metadata:
                        st.metric(
                            "Cache Hit Rate",
                            f"{metadata['cache_hit_rate']:.1f}%"
                        )
                    
                    # Throughput
                    if 'stage_throughput' in metadata:
                        st.metric(
                            "Processing Speed",
                            f"{metadata['stage_throughput']:.2f} items/sec"
                        )
                    
                    # Elapsed time
                    if self.start_time:
                        elapsed = time.time() - self.start_time
                        st.metric(
                            "Elapsed Time",
                            self._format_time(elapsed)
                        )
    
    def render_simple(self, message: str = "Loading...", progress: float = None):
        """
        Render a simple loading indicator without detailed progress
        
        Args:
            message: Loading message
            progress: Progress value 0-100 (optional)
        """
        with self.container:
            st.markdown(f"### ⏳ {message}")
            
            if progress is not None:
                st.progress(progress / 100 if progress <= 100 else 1.0)
                st.markdown(f"**Progress:** {progress:.1f}%")
            else:
                # Indeterminate progress
                st.markdown("*Processing...*")
    
    def complete(self, message: str = "Complete!", show_summary: bool = True):
        """
        Show completion message
        
        Args:
            message: Completion message
            show_summary: Whether to show summary statistics
        """
        with self.container:
            st.success(f"✅ {message}")
            
            if show_summary and self.start_time:
                elapsed = time.time() - self.start_time
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Time", self._format_time(elapsed))
                
                with col2:
                    if self.total_items:
                        rate = self.total_items / elapsed if elapsed > 0 else 0
                        st.metric("Processing Rate", f"{rate:.2f} items/sec")
                
                with col3:
                    st.metric("Items Processed", f"{self.total_items}")
    
    def error(self, message: str, details: str = None):
        """
        Show error message
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        with self.container:
            st.error(f"❌ {message}")
            
            if details:
                with st.expander("Error Details"):
                    st.code(details)
    
    def _is_stage_complete(self, check_stage: str, current_stage: str) -> bool:
        """Check if a stage is complete based on current stage"""
        stage_order = list(self.STAGES.keys())
        
        if check_stage not in stage_order or current_stage not in stage_order:
            return False
        
        return stage_order.index(check_stage) < stage_order.index(current_stage)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


# Example usage functions
def example_basic_usage():
    """Example: Basic loading indicator"""
    st.header("Example: Basic Loading")
    
    indicator = LoadingIndicator()
    
    if st.button("Start Basic Loading"):
        indicator.start(total_items=100)
        
        # Simulate progress
        for i in range(0, 101, 10):
            indicator.update({
                'stage': 'symbol_scanning',
                'current': i,
                'total': 100,
                'message': f'Processing item {i}/100...',
                'metadata': {
                    'overall_progress_pct': i,
                    'stage_progress_pct': i
                }
            })
            indicator.render()
            time.sleep(0.5)
        
        indicator.complete("Scan complete!")


def example_multi_stage():
    """Example: Multi-stage loading"""
    st.header("Example: Multi-Stage Loading")
    
    indicator = LoadingIndicator()
    
    if st.button("Start Multi-Stage Loading"):
        indicator.start(total_items=100)
        
        stages = [
            ('initializing', 5),
            ('universe_loading', 10),
            ('data_fetching', 30),
            ('symbol_scanning', 90),
            ('finalization', 100)
        ]
        
        for stage, progress in stages:
            indicator.update({
                'stage': stage,
                'current': progress,
                'total': 100,
                'message': f'Processing {stage}...',
                'metadata': {
                    'overall_progress_pct': progress,
                    'stage_progress_pct': (progress % 20) * 5,
                    'overall_eta': (100 - progress) * 0.5,
                    'cache_hit_rate': 75.5,
                    'stage_throughput': 2.5
                }
            })
            indicator.render()
            time.sleep(1)
        
        indicator.complete("All stages complete!")


def example_simple():
    """Example: Simple loading indicator"""
    st.header("Example: Simple Loading")
    
    indicator = LoadingIndicator()
    
    if st.button("Start Simple Loading"):
        for i in range(0, 101, 20):
            indicator.render_simple(
                message="Processing data...",
                progress=i
            )
            time.sleep(0.5)
        
        indicator.complete("Done!")


if __name__ == "__main__":
    st.set_page_config(page_title="Loading Indicator Examples", layout="wide")
    
    st.title("🔄 Loading Indicator Component")
    st.markdown("Examples of different loading indicator styles")
    
    tab1, tab2, tab3 = st.tabs(["Basic", "Multi-Stage", "Simple"])
    
    with tab1:
        example_basic_usage()
    
    with tab2:
        example_multi_stage()
    
    with tab3:
        example_simple()
