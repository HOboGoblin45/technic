"""
Test Redis connection with corrected URL format
"""

import os

# Try different URL formats based on your Redis Cloud setup
# Format 1: With password only (no username)
redis_url = 'redis://:ytvZ10VXoGV40enJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

print("Testing Redis connection...")
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
        print("\nTrying alternative connection method...")
        
        # Try direct connection with parameters
        import redis
        try:
            client = redis.Redis(
                host='redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com',
                port=12579,
                password='ytvZ10VXoGV40enJH3GJYkDLeg2emqad',
                db=0,
                socket_connect_timeout=10,
                decode_responses=False
            )
            
            # Test ping
            response = client.ping()
            print(f"✅ Direct connection successful! Ping response: {response}")
            
            # Test set/get
            client.set('test:key', 'test:value', ex=60)
            value = client.get('test:key')
            print(f"✅ Set/Get test successful! Value: {value}")
            
            # Cleanup
            client.delete('test:key')
            print("✅ Cleanup successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Direct connection also failed: {e}")
            return False
    
    print("\n✅ Redis is available!")
    
    # Test operations
    test_key = "technic:test"
    test_value = {"test": "data"}
    
    print(f"\nTesting set/get...")
    cache.set(test_key, test_value, ttl=60)
    retrieved = cache.get(test_key)
    
    if retrieved == test_value:
        print("✅ Set/Get test passed!")
        cache.delete(test_key)
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Set/Get test failed")
        return False

if __name__ == "__main__":
    success = test_redis()
    exit(0 if success else 1)
