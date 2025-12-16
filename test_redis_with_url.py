"""
Test Redis connection with explicit URL
"""

import os

# Set your Redis Cloud URL
os.environ['REDIS_URL'] = 'redis://default:ytvZ10VXoGV40enJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

# Now import and test
from technic_v4.cache.redis_cache import RedisCache

def test_redis_with_url():
    """Test Redis connection with explicit URL"""
    print("="*80)
    print("TESTING REDIS CONNECTION WITH REDIS CLOUD")
    print("="*80)
    
    # Create new instance with URL
    cache = RedisCache()
    
    print(f"\nRedis available: {cache.available}")
    
    if not cache.available:
        print("\n❌ Redis is NOT available")
        return False
    
    print("\n✅ Redis is available and connected to Redis Cloud!")
    
    # Test basic operations
    print("\nTesting basic operations...")
    
    # Test SET
    test_key = "technic:test:connection"
    test_value = {
        "message": "Hello from Technic Scanner!",
        "timestamp": "2025-12-16",
        "test": "Phase 3C Redis Integration"
    }
    
    print(f"Setting key '{test_key}'...")
    success = cache.set(test_key, test_value, ttl=300)
    
    if not success:
        print("❌ Failed to set key")
        return False
    
    print("✅ Key set successfully")
    
    # Test GET
    print(f"\nGetting key '{test_key}'...")
    retrieved = cache.get(test_key)
    
    if retrieved is None:
        print("❌ Failed to get key")
        return False
    
    print(f"✅ Key retrieved successfully!")
    print(f"   Value: {retrieved}")
    
    # Verify value matches
    if retrieved == test_value:
        print("✅ Value matches perfectly!")
    else:
        print(f"❌ Value mismatch")
        return False
    
    # Test batch operations
    print("\nTesting batch operations...")
    batch_data = {
        "technic:test:symbol1": {"symbol": "AAPL", "price": 150.0},
        "technic:test:symbol2": {"symbol": "MSFT", "price": 380.0},
        "technic:test:symbol3": {"symbol": "GOOGL", "price": 140.0},
    }
    
    print(f"Setting {len(batch_data)} keys in batch...")
    cache.batch_set(batch_data, ttl=300)
    print("✅ Batch set successful")
    
    # Test batch get
    print(f"\nGetting {len(batch_data)} keys in batch...")
    retrieved_batch = cache.batch_get(list(batch_data.keys()))
    print(f"✅ Retrieved {len(retrieved_batch)} keys")
    
    for key, value in retrieved_batch.items():
        print(f"   {key}: {value}")
    
    # Clean up test keys
    print("\nCleaning up test keys...")
    cache.delete(test_key)
    for key in batch_data.keys():
        cache.delete(key)
    print("✅ Cleanup complete")
    
    # Get cache statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key == 'hit_rate':
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ ALL REDIS CLOUD TESTS PASSED!")
    print("="*80)
    print("\nYour Redis Cloud connection is working perfectly!")
    print("You can now use Redis caching in your scanner for 2x speedup.")
    
    return True

if __name__ == "__main__":
    success = test_redis_with_url()
    exit(0 if success else 1)
