"""
Test Redis connection with your Redis Cloud credentials
"""

import os
from technic_v4.cache.redis_cache import redis_cache

def test_redis_connection():
    """Test that Redis connection works"""
    print("="*80)
    print("TESTING REDIS CONNECTION")
    print("="*80)
    
    # Check if Redis is available
    print(f"\nRedis available: {redis_cache.available}")
    
    if not redis_cache.available:
        print("\n❌ Redis is NOT available")
        print("Make sure REDIS_URL environment variable is set")
        return False
    
    print("\n✅ Redis is available!")
    
    # Test basic operations
    print("\nTesting basic operations...")
    
    # Test SET
    test_key = "test:connection"
    test_value = {"message": "Hello from Technic Scanner!", "timestamp": "2025-12-16"}
    
    print(f"Setting key '{test_key}'...")
    success = redis_cache.set(test_key, test_value, ttl=60)
    
    if not success:
        print("❌ Failed to set key")
        return False
    
    print("✅ Key set successfully")
    
    # Test GET
    print(f"\nGetting key '{test_key}'...")
    retrieved = redis_cache.get(test_key)
    
    if retrieved is None:
        print("❌ Failed to get key")
        return False
    
    print(f"✅ Key retrieved: {retrieved}")
    
    # Verify value matches
    if retrieved == test_value:
        print("✅ Value matches!")
    else:
        print(f"❌ Value mismatch: expected {test_value}, got {retrieved}")
        return False
    
    # Test DELETE
    print(f"\nDeleting key '{test_key}'...")
    redis_cache.delete(test_key)
    
    # Verify deletion
    retrieved_after_delete = redis_cache.get(test_key)
    if retrieved_after_delete is None:
        print("✅ Key deleted successfully")
    else:
        print("❌ Key still exists after deletion")
        return False
    
    # Get cache statistics
    print("\nCache Statistics:")
    stats = redis_cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ ALL REDIS TESTS PASSED!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_redis_connection()
    exit(0 if success else 1)
