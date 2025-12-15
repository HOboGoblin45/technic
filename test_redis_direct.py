import redis

# Test 1: Direct connection with host/port/password
print("=" * 80)
print("TEST 1: Direct connection (host, port, password)")
print("=" * 80)
try:
    r = redis.Redis(
        host='redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com',
        port=12579,
        password='ytvZ1OVXoGV40enJH3GJYkDLeg2emqad',
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    r.ping()
    print("✅ SUCCESS! Connection works!")
    print(f"Redis info: {r.info('server')['redis_version']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 2: Connection with username='default'
print("=" * 80)
print("TEST 2: Connection with username='default'")
print("=" * 80)
try:
    r = redis.Redis(
        host='redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com',
        port=12579,
        username='default',
        password='ytvZ1OVXoGV40enJH3GJYkDLeg2emqad',
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    r.ping()
    print("✅ SUCCESS! Connection works!")
    print(f"Redis info: {r.info('server')['redis_version']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 3: Connection WITHOUT username
print("=" * 80)
print("TEST 3: Connection WITHOUT username (password only)")
print("=" * 80)
try:
    r = redis.Redis(
        host='redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com',
        port=12579,
        password='ytvZ1OVXoGV40enJH3GJYkDLeg2emqad',
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    r.ping()
    print("✅ SUCCESS! Connection works!")
    
    # Test set/get
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    print(f"✅ Set/Get test: {value}")
    
    # Clean up
    r.delete('test_key')
    print("✅ Redis is fully functional!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 4: URL connection
print("=" * 80)
print("TEST 4: URL connection")
print("=" * 80)
try:
    r = redis.from_url(
        'redis://default:ytvZ1OVXoGV40enJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0',
        decode_responses=True,
        socket_connect_timeout=5
    )
    r.ping()
    print("✅ SUCCESS! Connection works!")
except Exception as e:
    print(f"❌ FAILED: {e}")
