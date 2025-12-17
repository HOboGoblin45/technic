"""
Scan History Database
Stores and retrieves historical scan data for ML training
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class ScanRecord:
    """Record of a completed scan"""
    scan_id: str
    timestamp: datetime
    config: Dict[str, Any]  # ScanConfig as dict
    results: Dict[str, Any]  # Results summary
    performance: Dict[str, Any]  # Timing, throughput
    market_conditions: Dict[str, Any]  # SPY data, VIX, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'scan_id': self.scan_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'config': self.config,
            'results': self.results,
            'performance': self.performance,
            'market_conditions': self.market_conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScanRecord':
        """Create from dictionary"""
        # Parse timestamp
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            scan_id=data['scan_id'],
            timestamp=timestamp,
            config=data['config'],
            results=data['results'],
            performance=data['performance'],
            market_conditions=data['market_conditions']
        )


class ScanHistoryDB:
    """
    Store and retrieve scan history using JSONL format
    
    Features:
    - Append-only for performance
    - Query by config parameters
    - Get recent scans
    - Calculate statistics
    """
    
    def __init__(self, db_path: str = "data/scan_history.jsonl"):
        """
        Initialize scan history database
        
        Args:
            db_path: Path to JSONL database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file if it doesn't exist
        if not self.db_path.exists():
            self.db_path.touch()
    
    def add_scan(self, record: ScanRecord):
        """
        Add scan record to database
        
        Args:
            record: ScanRecord to add
        """
        with open(self.db_path, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')
    
    def get_recent_scans(self, limit: int = 100) -> List[ScanRecord]:
        """
        Get recent scan records
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of recent ScanRecord objects
        """
        records = []
        
        if not self.db_path.exists():
            return records
        
        # Read file in reverse to get most recent first
        with open(self.db_path, 'r') as f:
            lines = f.readlines()
        
        # Parse most recent records
        for line in reversed(lines[-limit:]):
            if line.strip():
                try:
                    data = json.loads(line)
                    records.append(ScanRecord.from_dict(data))
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def get_scans_by_config(self, config_filter: Dict[str, Any]) -> List[ScanRecord]:
        """
        Get scans matching config criteria
        
        Args:
            config_filter: Dictionary of config parameters to match
        
        Returns:
            List of matching ScanRecord objects
        """
        records = []
        
        if not self.db_path.exists():
            return records
        
        with open(self.db_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    record = ScanRecord.from_dict(data)
                    
                    # Check if config matches filter
                    if self._matches_filter(record.config, config_filter):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def get_scans_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ScanRecord]:
        """
        Get scans within date range
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            List of ScanRecord objects in range
        """
        records = []
        
        if not self.db_path.exists():
            return records
        
        with open(self.db_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    record = ScanRecord.from_dict(data)
                    
                    if start_date <= record.timestamp <= end_date:
                        records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.db_path.exists():
            return {
                'total_scans': 0,
                'date_range': None,
                'avg_results': 0,
                'avg_duration': 0
            }
        
        records = self.get_recent_scans(limit=1000)
        
        if not records:
            return {
                'total_scans': 0,
                'date_range': None,
                'avg_results': 0,
                'avg_duration': 0
            }
        
        # Calculate statistics
        total_results = sum(r.results.get('count', 0) for r in records)
        total_duration = sum(r.performance.get('total_seconds', 0) for r in records)
        
        return {
            'total_scans': len(records),
            'date_range': {
                'start': min(r.timestamp for r in records).isoformat(),
                'end': max(r.timestamp for r in records).isoformat()
            },
            'avg_results': total_results / len(records) if records else 0,
            'avg_duration': total_duration / len(records) if records else 0,
            'total_results': total_results,
            'total_duration': total_duration
        }
    
    def _matches_filter(self, config: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if config matches filter
        
        Args:
            config: Configuration dictionary
            filter_dict: Filter criteria
        
        Returns:
            True if config matches all filter criteria
        """
        for key, value in filter_dict.items():
            if key not in config:
                return False
            
            # Handle list comparisons (e.g., sectors)
            if isinstance(value, list) and isinstance(config[key], list):
                if not set(value).issubset(set(config[key])):
                    return False
            # Handle exact matches
            elif config[key] != value:
                return False
        
        return True
    
    def clear_old_records(self, days: int = 90):
        """
        Remove records older than specified days
        
        Args:
            days: Number of days to keep
        """
        if not self.db_path.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Read all records
        records = []
        with open(self.db_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    record = ScanRecord.from_dict(data)
                    
                    if record.timestamp >= cutoff_date:
                        records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Rewrite file with recent records only
        with open(self.db_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + '\n')


# Import for timedelta
from datetime import timedelta


if __name__ == "__main__":
    # Test the scan history database
    print("Testing Scan History Database...")
    
    # Create database
    db = ScanHistoryDB("data/test_scan_history.jsonl")
    
    # Create test record
    test_record = ScanRecord(
        scan_id="test_001",
        timestamp=datetime.now(),
        config={
            'max_symbols': 100,
            'min_tech_rating': 30,
            'sectors': ['Technology', 'Healthcare']
        },
        results={
            'count': 15,
            'signals': 5
        },
        performance={
            'total_seconds': 12.5,
            'symbols_per_second': 8.0
        },
        market_conditions={
            'spy_trend': 'bullish',
            'spy_volatility': 0.15
        }
    )
    
    # Add record
    db.add_scan(test_record)
    print("✓ Added test record")
    
    # Get recent scans
    recent = db.get_recent_scans(limit=10)
    print(f"✓ Retrieved {len(recent)} recent scans")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"✓ Database statistics: {stats}")
    
    print("\n✓ Scan History Database test complete!")
