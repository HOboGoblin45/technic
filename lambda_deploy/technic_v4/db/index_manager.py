"""
Database Index Manager
Manages database indexes for optimal query performance
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime


@dataclass
class IndexDefinition:
    """Definition of a database index"""
    name: str
    table: str
    columns: List[str]
    index_type: str = "btree"  # btree, hash, gin, gist
    unique: bool = False
    partial_condition: Optional[str] = None
    created: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'table': self.table,
            'columns': self.columns,
            'index_type': self.index_type,
            'unique': self.unique,
            'partial_condition': self.partial_condition,
            'created': self.created
        }
    
    def get_create_statement(self) -> str:
        """Generate CREATE INDEX statement"""
        unique_str = "UNIQUE " if self.unique else ""
        columns_str = ", ".join(self.columns)
        
        stmt = f"CREATE {unique_str}INDEX {self.name} ON {self.table}"
        
        if self.index_type != "btree":
            stmt += f" USING {self.index_type}"
        
        stmt += f" ({columns_str})"
        
        if self.partial_condition:
            stmt += f" WHERE {self.partial_condition}"
        
        return stmt + ";"


class IndexManager:
    """
    Manages database indexes for query optimization
    
    Features:
    - Strategic index recommendations
    - Index creation and management
    - Index usage analysis
    - Performance impact tracking
    
    Example:
        >>> manager = IndexManager()
        >>> manager.recommend_indexes_for_query("SELECT * FROM symbols WHERE sector = 'Technology'")
        >>> manager.create_recommended_indexes()
        >>> manager.analyze_index_usage()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize index manager
        
        Args:
            config_path: Path to index configuration file
        """
        self.indexes: Dict[str, IndexDefinition] = {}
        self.config_path = config_path or "config/indexes.json"
        self.load_config()
    
    def add_index(self, index: IndexDefinition):
        """
        Add an index definition
        
        Args:
            index: Index definition to add
        """
        self.indexes[index.name] = index
        print(f"[INDEX] Added index definition: {index.name}")
    
    def recommend_indexes_for_scanner(self) -> List[IndexDefinition]:
        """
        Recommend indexes for scanner queries
        
        Returns:
            List of recommended index definitions
        """
        recommendations = []
        
        # Index 1: Symbol lookup (most common query)
        recommendations.append(IndexDefinition(
            name="idx_symbols_symbol",
            table="symbols",
            columns=["symbol"],
            unique=True,
            index_type="hash"  # Hash index for exact matches
        ))
        
        # Index 2: Sector filtering
        recommendations.append(IndexDefinition(
            name="idx_symbols_sector",
            table="symbols",
            columns=["sector"],
            index_type="btree"
        ))
        
        # Index 3: Industry filtering
        recommendations.append(IndexDefinition(
            name="idx_symbols_industry",
            table="symbols",
            columns=["industry"],
            index_type="btree"
        ))
        
        # Index 4: Composite index for sector + industry
        recommendations.append(IndexDefinition(
            name="idx_symbols_sector_industry",
            table="symbols",
            columns=["sector", "industry"],
            index_type="btree"
        ))
        
        # Index 5: Market cap filtering
        recommendations.append(IndexDefinition(
            name="idx_symbols_market_cap",
            table="symbols",
            columns=["market_cap"],
            index_type="btree"
        ))
        
        # Index 6: Active symbols only (partial index)
        recommendations.append(IndexDefinition(
            name="idx_symbols_active",
            table="symbols",
            columns=["symbol", "sector"],
            partial_condition="active = true"
        ))
        
        # Index 7: Price history by symbol and date
        recommendations.append(IndexDefinition(
            name="idx_price_history_symbol_date",
            table="price_history",
            columns=["symbol", "date"],
            index_type="btree"
        ))
        
        # Index 8: Recent price history (partial index)
        recommendations.append(IndexDefinition(
            name="idx_price_history_recent",
            table="price_history",
            columns=["symbol", "date", "close"],
            partial_condition="date >= CURRENT_DATE - INTERVAL '90 days'"
        ))
        
        # Index 9: Scan results by date
        recommendations.append(IndexDefinition(
            name="idx_scan_results_date",
            table="scan_results",
            columns=["scan_date", "symbol"],
            index_type="btree"
        ))
        
        # Index 10: High-scoring results (partial index)
        recommendations.append(IndexDefinition(
            name="idx_scan_results_high_score",
            table="scan_results",
            columns=["scan_date", "tech_rating", "alpha_score"],
            partial_condition="tech_rating >= 70"
        ))
        
        for idx in recommendations:
            self.add_index(idx)
        
        return recommendations
    
    def recommend_indexes_for_query(self, query: str) -> List[IndexDefinition]:
        """
        Analyze a query and recommend indexes
        
        Args:
            query: SQL query to analyze
        
        Returns:
            List of recommended indexes
        """
        recommendations = []
        query_lower = query.lower()
        
        # Simple heuristics for index recommendations
        if "where" in query_lower:
            # Extract WHERE conditions (simplified)
            where_clause = query_lower.split("where")[1].split("order by")[0] if "order by" in query_lower else query_lower.split("where")[1]
            
            # Look for common patterns
            if "symbol =" in where_clause or "symbol in" in where_clause:
                recommendations.append(IndexDefinition(
                    name="idx_query_symbol",
                    table="inferred_table",
                    columns=["symbol"],
                    index_type="hash"
                ))
            
            if "sector =" in where_clause:
                recommendations.append(IndexDefinition(
                    name="idx_query_sector",
                    table="inferred_table",
                    columns=["sector"]
                ))
            
            if "date >=" in where_clause or "date between" in where_clause:
                recommendations.append(IndexDefinition(
                    name="idx_query_date",
                    table="inferred_table",
                    columns=["date"]
                ))
        
        if "order by" in query_lower:
            # Extract ORDER BY columns
            order_clause = query_lower.split("order by")[1].split("limit")[0] if "limit" in query_lower else query_lower.split("order by")[1]
            # Could recommend index on ORDER BY columns
        
        return recommendations
    
    def create_index(self, index_name: str) -> bool:
        """
        Create an index (placeholder - actual implementation depends on database)
        
        Args:
            index_name: Name of index to create
        
        Returns:
            True if successful
        """
        if index_name not in self.indexes:
            print(f"[INDEX] Index {index_name} not found")
            return False
        
        index = self.indexes[index_name]
        
        if index.created:
            print(f"[INDEX] Index {index_name} already created")
            return True
        
        # Generate CREATE INDEX statement
        create_stmt = index.get_create_statement()
        print(f"[INDEX] Would execute: {create_stmt}")
        
        # In a real implementation, execute the statement
        # For now, just mark as created
        index.created = True
        
        print(f"[INDEX] Created index: {index_name}")
        return True
    
    def create_all_indexes(self) -> Dict[str, bool]:
        """
        Create all defined indexes
        
        Returns:
            Dictionary of index_name -> success status
        """
        results = {}
        
        for index_name in self.indexes:
            results[index_name] = self.create_index(index_name)
        
        return results
    
    def drop_index(self, index_name: str) -> bool:
        """
        Drop an index
        
        Args:
            index_name: Name of index to drop
        
        Returns:
            True if successful
        """
        if index_name not in self.indexes:
            print(f"[INDEX] Index {index_name} not found")
            return False
        
        index = self.indexes[index_name]
        
        if not index.created:
            print(f"[INDEX] Index {index_name} not created yet")
            return True
        
        drop_stmt = f"DROP INDEX IF EXISTS {index_name};"
        print(f"[INDEX] Would execute: {drop_stmt}")
        
        # In a real implementation, execute the statement
        index.created = False
        
        print(f"[INDEX] Dropped index: {index_name}")
        return True
    
    def analyze_index_usage(self) -> Dict[str, Dict]:
        """
        Analyze index usage statistics
        
        Returns:
            Dictionary of index statistics
        """
        stats = {}
        
        for index_name, index in self.indexes.items():
            stats[index_name] = {
                'created': index.created,
                'table': index.table,
                'columns': index.columns,
                'type': index.index_type,
                'unique': index.unique,
                'partial': index.partial_condition is not None,
                # In real implementation, would query database for:
                # - scans using this index
                # - index size
                # - last used timestamp
                'estimated_benefit': 'high' if 'symbol' in index.columns else 'medium'
            }
        
        return stats
    
    def get_index_recommendations_report(self) -> str:
        """
        Generate a report of index recommendations
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("INDEX RECOMMENDATIONS REPORT")
        report.append("="*80)
        report.append("")
        
        if not self.indexes:
            report.append("No indexes defined yet.")
            report.append("Run recommend_indexes_for_scanner() to generate recommendations.")
            return "\n".join(report)
        
        report.append(f"Total Indexes: {len(self.indexes)}")
        report.append(f"Created: {sum(1 for idx in self.indexes.values() if idx.created)}")
        report.append(f"Pending: {sum(1 for idx in self.indexes.values() if not idx.created)}")
        report.append("")
        
        # Group by table
        by_table = {}
        for idx in self.indexes.values():
            if idx.table not in by_table:
                by_table[idx.table] = []
            by_table[idx.table].append(idx)
        
        for table, indexes in sorted(by_table.items()):
            report.append(f"Table: {table}")
            report.append("-" * 80)
            
            for idx in indexes:
                status = "✓" if idx.created else "○"
                columns_str = ", ".join(idx.columns)
                report.append(f"  {status} {idx.name}")
                report.append(f"     Columns: {columns_str}")
                report.append(f"     Type: {idx.index_type}")
                if idx.unique:
                    report.append(f"     Unique: Yes")
                if idx.partial_condition:
                    report.append(f"     Partial: {idx.partial_condition}")
                report.append("")
        
        report.append("="*80)
        return "\n".join(report)
    
    def save_config(self):
        """Save index configuration to file"""
        config = {
            'last_updated': datetime.now().isoformat(),
            'indexes': {
                name: idx.to_dict()
                for name, idx in self.indexes.items()
            }
        }
        
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[INDEX] Saved configuration to {self.config_path}")
    
    def load_config(self):
        """Load index configuration from file"""
        if not Path(self.config_path).exists():
            print(f"[INDEX] No configuration file found at {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for name, idx_data in config.get('indexes', {}).items():
                index = IndexDefinition(
                    name=idx_data['name'],
                    table=idx_data['table'],
                    columns=idx_data['columns'],
                    index_type=idx_data.get('index_type', 'btree'),
                    unique=idx_data.get('unique', False),
                    partial_condition=idx_data.get('partial_condition'),
                    created=idx_data.get('created', False)
                )
                self.indexes[name] = index
            
            print(f"[INDEX] Loaded {len(self.indexes)} indexes from {self.config_path}")
        
        except Exception as e:
            print(f"[INDEX] Error loading configuration: {e}")


# Global index manager instance
_global_index_manager = IndexManager()


def get_index_manager() -> IndexManager:
    """Get the global index manager instance"""
    return _global_index_manager


if __name__ == "__main__":
    # Example usage
    manager = IndexManager()
    
    # Generate recommendations
    print("Generating index recommendations for scanner...")
    recommendations = manager.recommend_indexes_for_scanner()
    
    # Print report
    print(manager.get_index_recommendations_report())
    
    # Save configuration
    manager.save_config()
    
    # Analyze usage
    stats = manager.analyze_index_usage()
    print("\nIndex Usage Statistics:")
    for idx_name, idx_stats in stats.items():
        print(f"  {idx_name}: {idx_stats}")
