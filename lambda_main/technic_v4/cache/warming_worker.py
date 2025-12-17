"""
Background Cache Warming Worker
Automated cache warming with scheduling and monitoring
Path 1 Task 6: Smart Cache Warming
"""

import time
import threading
import schedule
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

from technic_v4.cache.cache_warmer import SmartCacheWarmer, WarmingConfig, get_warmer
from technic_v4.cache.access_tracker import AccessPatternTracker, get_tracker


logger = logging.getLogger(__name__)


class CacheWarmingWorker:
    """
    Background worker for automated cache warming
    
    Features:
    - Scheduled warming cycles
    - Market hours awareness
    - Resource monitoring
    - Graceful shutdown
    - Performance tracking
    
    Example:
        >>> worker = CacheWarmingWorker()
        >>> worker.start()
        >>> # ... worker runs in background ...
        >>> worker.stop()
    """
    
    def __init__(
        self,
        warmer: Optional[SmartCacheWarmer] = None,
        tracker: Optional[AccessPatternTracker] = None
    ):
        """
        Initialize warming worker
        
        Args:
            warmer: Cache warmer instance
            tracker: Access pattern tracker
        """
        self.warmer = warmer or get_warmer()
        self.tracker = tracker or get_tracker()
        
        # Worker state
        self.running = False
        self.thread = None
        self.last_cycle = None
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_history = []
        self.max_history = 100
        
        # Setup schedules
        self._setup_schedules()
    
    def _setup_schedules(self):
        """Setup warming schedules"""
        # Popular symbols every 30 minutes
        schedule.every(30).minutes.do(self._warm_popular)
        
        # Time-based patterns every hour
        schedule.every(1).hours.do(self._warm_time_based)
        
        # Trending symbols every hour
        schedule.every(1).hours.do(self._warm_trending)
        
        # Pre-market warming at 8:30 AM
        schedule.every().day.at("08:30").do(self._warm_pre_market)
        
        # Save tracker data every 15 minutes
        schedule.every(15).minutes.do(self._save_tracker_data)
        
        # Cleanup old data daily at midnight
        schedule.every().day.at("00:00").do(self._cleanup_old_data)
    
    def _warm_popular(self):
        """Warm popular symbols"""
        try:
            logger.info("[WORKER] Running popular symbols warming...")
            result = self.warmer.warm_popular_symbols()
            
            self._record_performance('popular', result)
            
            logger.info(
                f"[WORKER] Popular warming complete: "
                f"{result.symbols_warmed} warmed, {result.symbols_failed} failed"
            )
        except Exception as e:
            logger.error(f"[WORKER] Popular warming failed: {e}", exc_info=True)
    
    def _warm_time_based(self):
        """Warm based on time patterns"""
        try:
            logger.info("[WORKER] Running time-based warming...")
            result = self.warmer.warm_by_time_pattern(look_ahead_hours=1)
            
            self._record_performance('time_based', result)
            
            logger.info(
                f"[WORKER] Time-based warming complete: "
                f"{result.symbols_warmed} warmed"
            )
        except Exception as e:
            logger.error(f"[WORKER] Time-based warming failed: {e}", exc_info=True)
    
    def _warm_trending(self):
        """Warm trending symbols"""
        try:
            logger.info("[WORKER] Running trending symbols warming...")
            result = self.warmer.warm_trending(window_hours=24)
            
            self._record_performance('trending', result)
            
            logger.info(
                f"[WORKER] Trending warming complete: "
                f"{result.symbols_warmed} warmed"
            )
        except Exception as e:
            logger.error(f"[WORKER] Trending warming failed: {e}", exc_info=True)
    
    def _warm_pre_market(self):
        """Pre-market warming (8:30 AM)"""
        try:
            logger.info("[WORKER] Running pre-market warming...")
            
            # Warm top 200 popular symbols before market open
            result = self.warmer.warm_popular_symbols(limit=200)
            
            self._record_performance('pre_market', result)
            
            logger.info(
                f"[WORKER] Pre-market warming complete: "
                f"{result.symbols_warmed} warmed in {result.duration_seconds:.1f}s"
            )
        except Exception as e:
            logger.error(f"[WORKER] Pre-market warming failed: {e}", exc_info=True)
    
    def _save_tracker_data(self):
        """Save access tracker data"""
        try:
            self.tracker.save_data()
            logger.debug("[WORKER] Tracker data saved")
        except Exception as e:
            logger.error(f"[WORKER] Failed to save tracker data: {e}")
    
    def _cleanup_old_data(self):
        """Cleanup old access data"""
        try:
            self.tracker.clear_old_data(days=30)
            logger.info("[WORKER] Old data cleaned up")
        except Exception as e:
            logger.error(f"[WORKER] Failed to cleanup old data: {e}")
    
    def _record_performance(self, strategy: str, result):
        """Record warming performance"""
        record = {
            'timestamp': time.time(),
            'strategy': strategy,
            'symbols_warmed': result.symbols_warmed,
            'symbols_failed': result.symbols_failed,
            'duration': result.duration_seconds,
            'success_rate': (
                result.symbols_warmed / (result.symbols_warmed + result.symbols_failed)
                if (result.symbols_warmed + result.symbols_failed) > 0
                else 0
            )
        }
        
        self.performance_history.append(record)
        
        # Keep only recent history
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
    
    def _run_loop(self):
        """Main worker loop"""
        logger.info("[WORKER] Cache warming worker started")
        
        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Update cycle info
                self.last_cycle = datetime.now()
                self.cycle_count += 1
                
                # Sleep for 1 second
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"[WORKER] Error in worker loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
        
        logger.info("[WORKER] Cache warming worker stopped")
    
    def start(self):
        """Start the background worker"""
        if self.running:
            logger.warning("[WORKER] Worker already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        logger.info("[WORKER] Cache warming worker thread started")
    
    def stop(self):
        """Stop the background worker"""
        if not self.running:
            logger.warning("[WORKER] Worker not running")
            return
        
        logger.info("[WORKER] Stopping cache warming worker...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        # Save final data
        try:
            self.tracker.save_data()
        except Exception as e:
            logger.error(f"[WORKER] Failed to save final data: {e}")
        
        logger.info("[WORKER] Cache warming worker stopped")
    
    def get_status(self) -> Dict:
        """Get worker status"""
        return {
            'running': self.running,
            'last_cycle': self.last_cycle.isoformat() if self.last_cycle else None,
            'cycle_count': self.cycle_count,
            'next_runs': self._get_next_runs(),
            'performance_summary': self._get_performance_summary()
        }
    
    def _get_next_runs(self) -> Dict[str, str]:
        """Get next scheduled run times"""
        next_runs = {}
        
        for job in schedule.jobs:
            # Get job name from tags or function name
            job_name = (
                list(job.tags)[0] if job.tags
                else job.job_func.__name__
            )
            
            next_run = job.next_run
            if next_run:
                next_runs[job_name] = next_run.isoformat()
        
        return next_runs
    
    def _get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.performance_history:
            return {}
        
        # Calculate averages
        total_warmed = sum(r['symbols_warmed'] for r in self.performance_history)
        total_failed = sum(r['symbols_failed'] for r in self.performance_history)
        avg_duration = sum(r['duration'] for r in self.performance_history) / len(self.performance_history)
        avg_success_rate = sum(r['success_rate'] for r in self.performance_history) / len(self.performance_history)
        
        # By strategy
        by_strategy = {}
        for record in self.performance_history:
            strategy = record['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = {
                    'count': 0,
                    'total_warmed': 0,
                    'total_failed': 0
                }
            
            by_strategy[strategy]['count'] += 1
            by_strategy[strategy]['total_warmed'] += record['symbols_warmed']
            by_strategy[strategy]['total_failed'] += record['symbols_failed']
        
        return {
            'total_cycles': len(self.performance_history),
            'total_warmed': total_warmed,
            'total_failed': total_failed,
            'avg_duration_seconds': round(avg_duration, 2),
            'avg_success_rate': round(avg_success_rate * 100, 1),
            'by_strategy': by_strategy
        }
    
    def run_manual_cycle(self):
        """Manually trigger a warming cycle"""
        logger.info("[WORKER] Running manual warming cycle...")
        
        results = self.warmer.run_all_strategies()
        
        for strategy, result in results.items():
            self._record_performance(strategy, result)
            logger.info(
                f"[WORKER] {strategy}: {result.symbols_warmed} warmed, "
                f"{result.symbols_failed} failed"
            )
        
        return results


# Global worker instance
_global_worker = None


def get_worker() -> CacheWarmingWorker:
    """Get global warming worker instance"""
    global _global_worker
    if _global_worker is None:
        _global_worker = CacheWarmingWorker()
    return _global_worker


def start_warming_worker():
    """Start the global warming worker"""
    worker = get_worker()
    worker.start()
    return worker


def stop_warming_worker():
    """Stop the global warming worker"""
    worker = get_worker()
    worker.stop()


if __name__ == "__main__":
    # Example usage
    import json
    
    print("Starting cache warming worker...")
    print("=" * 60)
    
    worker = CacheWarmingWorker()
    worker.start()
    
    print("Worker started. Running for 60 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        # Run for 60 seconds
        for i in range(60):
            time.sleep(1)
            
            if i % 10 == 0:
                status = worker.get_status()
                print(f"\n[{i}s] Worker Status:")
                print(f"  Running: {status['running']}")
                print(f"  Cycles: {status['cycle_count']}")
                print(f"  Last Cycle: {status['last_cycle']}")
    
    except KeyboardInterrupt:
        print("\n\nStopping worker...")
    
    finally:
        worker.stop()
        
        # Show final status
        status = worker.get_status()
        print("\nFinal Status:")
        print(json.dumps(status, indent=2))
