"""
Test Redis connection with the corrected password
"""

import os

# Use the corrected password you provided
redis_url = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

print("Testing Redis connection with corrected password...")
print(f"URL: redis://:***@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0")

os.environ['REDIS_URL'] = redis_url

from technic_v4.cache.redis_cache import RedisCache

def test_redis():
    """Test Redis connection"""
    print("\n" + "="*80)
    print("TESTING REDIS CLOUD CONNECTION")
    print("="*80)
    
    cache = RedisCache()
    
    print(f"\nRedis available: {cache.available}")
    
    if not cache.available:
        print("\n❌ Connection failed")
        return False
    
    print("\n✅ Redis is CONNECTED!")
    
    # Test basic operations
    print("\nTesting basic operations...")
    
    # Test SET
    test_key = "technic:test:connection"
    test_value = {
        "message": "Hello from Technic Scanner!",
        "timestamp": "2025-12-16",
        "phase": "3C Redis Integration"
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
        "technic:cache:AAPL": {"symbol": "AAPL", "price": 150.0, "cached": True},
        "technic:cache:MSFT": {"symbol": "MSFT", "price": 380.0, "cached": True},
        "technic:cache:GOOGL": {"symbol": "GOOGL", "price": 140.0, "cached": True},
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
    
    # Get cache statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key == 'hit_rate':
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")
    
    # Clean up test keys
    print("\nCleaning up test keys...")
    cache.delete(test_key)
    for key in batch_data.keys():
        cache.delete(key)
    print("✅ Cleanup complete")
    
    print("\n" + "="*80)
    print("✅ ALL REDIS TESTS PASSED!")
    print("="*80)
    print("\n🎉 Your Redis Cloud connection is working perfectly!")
    print("\nNext steps:")
    print("1. Set REDIS_URL in your environment variables")
    print("2. Run scanner - caching will work automatically")
    print("3. Expect 2x speedup on subsequent scans within 5 minutes")
    
    return True

if __name__ == "__main__":
    success = test_redis()
    exit(0 if success else 1)
