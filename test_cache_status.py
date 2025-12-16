"""
Test Cache Status Display - Phase 3D-A Task 2
Tests the enhanced cache statistics functionality
"""

import os
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

from technic_v4.cache.redis_cache import redis_cache
import json

print("="*80)
print("CACHE STATUS DISPLAY TEST")
print("="*80)

# Test 1: Basic Stats
print("\n" + "="*80)
print("TEST 1: Basic Cache Statistics")
print("="*80)

stats = redis_cache.get_stats()
print("\nBasic Stats:")
print(json.dumps(stats, indent=2))

if stats.get('available'):
    print("\n✅ Cache is available")
    print(f"   Total Keys: {stats.get('total_keys', 0)}")
    print(f"   Hit Rate: {stats.get('hit_rate', 0):.2f}%")
    print(f"   Hits: {stats.get('hits', 0)}")
    print(f"   Misses: {stats.get('misses', 0)}")
else:
    print("\n❌ Cache is not available")

# Test 2: Detailed Stats
print("\n" + "="*80)
print("TEST 2: Detailed Cache Statistics")
print("="*80)

detailed_stats = redis_cache.get_detailed_stats()
print("\nDetailed Stats:")
print(json.dumps(detailed_stats, indent=2, default=str))

if detailed_stats.get('available'):
    print("\n✅ Detailed stats retrieved successfully")
    
    # Connection info
    if 'connection' in detailed_stats:
        conn = detailed_stats['connection']
        print(f"\nConnection:")
        print(f"   Host: {conn.get('host', 'unknown')}")
        print(f"   Port: {conn.get('port', 0)}")
        print(f"   DB: {conn.get('db', 0)}")
    
    # Performance metrics
    if 'performance' in detailed_stats:
        perf = detailed_stats['performance']
        print(f"\nPerformance:")
        print(f"   Total Keys: {perf.get('total_keys', 0)}")
        print(f"   Hit Rate: {perf.get('hit_rate', 0):.2f}%")
        print(f"   Total Requests: {perf.get('total_requests', 0)}")
    
    # Memory usage
    if 'memory' in detailed_stats:
        mem = detailed_stats['memory']
        print(f"\nMemory:")
        print(f"   Used: {mem.get('used_mb', 0):.2f} MB")
        print(f"   Peak: {mem.get('peak_mb', 0):.2f} MB")
        print(f"   Fragmentation: {mem.get('fragmentation_ratio', 0):.2f}")
    
    # Keys by type
    if 'keys_by_type' in detailed_stats:
        keys_by_type = detailed_stats['keys_by_type']
        if keys_by_type:
            print(f"\nKeys by Type:")
            for key_type, count in sorted(keys_by_type.items(), key=lambda x: -x[1]):
                print(f"   {key_type}: {count}")
    
    # Server info
    if 'server_info' in detailed_stats:
        server = detailed_stats['server_info']
        print(f"\nServer Info:")
        print(f"   Redis Version: {server.get('redis_version', 'unknown')}")
        uptime = server.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        print(f"   Uptime: {uptime_hours:.1f} hours")
else:
    print("\n❌ Detailed stats not available")
    if 'error' in detailed_stats:
        print(f"   Error: {detailed_stats['error']}")

# Test 3: Cache Operations
print("\n" + "="*80)
print("TEST 3: Cache Operations & Stats Update")
print("="*80)

if redis_cache.available:
    # Perform some cache operations
    print("\nPerforming cache operations...")
    
    # Set some test data
    test_data = {
        'test:key1': 'value1',
        'test:key2': 'value2',
        'test:key3': 'value3'
    }
    
    redis_cache.batch_set(test_data, ttl=60)
    print(f"✅ Set {len(test_data)} test keys")
    
    # Get some data (hits)
    for key in test_data.keys():
        value = redis_cache.get(key)
        if value:
            print(f"✅ Cache hit for {key}")
    
    # Try to get non-existent key (miss)
    redis_cache.get('test:nonexistent')
    print("✅ Cache miss for nonexistent key")
    
    # Get updated stats
    print("\nUpdated Stats:")
    updated_stats = redis_cache.get_stats()
    print(f"   Total Keys: {updated_stats.get('total_keys', 0)}")
    print(f"   Hit Rate: {updated_stats.get('hit_rate', 0):.2f}%")
    
    # Clean up test keys
    redis_cache.clear_pattern('test:*')
    print("\n✅ Cleaned up test keys")
else:
    print("\n⚠️  Cache not available, skipping operations test")

# Test 4: Cache Status for Frontend Display
print("\n" + "="*80)
print("TEST 4: Frontend-Ready Cache Status")
print("="*80)

def get_cache_status_for_frontend():
    """Format cache status for frontend display"""
    stats = redis_cache.get_detailed_stats()
    
    if not stats.get('available'):
        return {
            'status': 'unavailable',
            'message': 'Cache is not available',
            'badge': '⚠️ No Cache'
        }
    
    perf = stats.get('performance', {})
    mem = stats.get('memory', {})
    hit_rate = perf.get('hit_rate', 0)
    
    # Determine status badge
    if hit_rate >= 80:
        badge = f'⚡ Cache Active ({hit_rate:.1f}% hit rate)'
        status = 'excellent'
    elif hit_rate >= 50:
        badge = f'✓ Cache Active ({hit_rate:.1f}% hit rate)'
        status = 'good'
    else:
        badge = f'⚠️ Cache Active ({hit_rate:.1f}% hit rate)'
        status = 'poor'
    
    return {
        'status': status,
        'badge': badge,
        'metrics': {
            'hit_rate': hit_rate,
            'total_keys': perf.get('total_keys', 0),
            'memory_used_mb': mem.get('used_mb', 0),
            'total_requests': perf.get('total_requests', 0)
        },
        'connection': stats.get('connection', {}),
        'keys_by_type': stats.get('keys_by_type', {})
    }

frontend_status = get_cache_status_for_frontend()
print("\nFrontend-Ready Status:")
print(json.dumps(frontend_status, indent=2))

print(f"\nStatus Badge: {frontend_status['badge']}")
print(f"Overall Status: {frontend_status['status']}")

if frontend_status['status'] != 'unavailable':
    metrics = frontend_status['metrics']
    print(f"\nKey Metrics:")
    print(f"   Hit Rate: {metrics['hit_rate']:.1f}%")
    print(f"   Total Keys: {metrics['total_keys']}")
    print(f"   Memory Used: {metrics['memory_used_mb']:.2f} MB")
    print(f"   Total Requests: {metrics['total_requests']}")
    
    print("\n✅ Cache status ready for frontend display")
else:
    print("\n⚠️  Cache unavailable")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

test_results = {
    'basic_stats': stats.get('available', False),
    'detailed_stats': detailed_stats.get('available', False),
    'cache_operations': redis_cache.available,
    'frontend_ready': frontend_status['status'] != 'unavailable'
}

print("\nTest Results:")
for test_name, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {test_name.replace('_', ' ').title()}")

passed_count = sum(test_results.values())
total_count = len(test_results)
pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

print(f"\nOverall: {passed_count}/{total_count} tests passed ({pass_rate:.1f}%)")

if pass_rate >= 75:
    print("\n✅ CACHE STATUS DISPLAY READY FOR INTEGRATION!")
else:
    print(f"\n⚠️  Some tests failed - review results above")

print("\n" + "="*80)
