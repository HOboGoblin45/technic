"""
Test both password variants to find the correct one
"""

import redis

passwords = [
    ("Original (from first screenshot)", "ytvZ10VXoGV40enJH3GJYkDLeg2emqad"),
    ("Working (from test)", "ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad"),
]

host = "redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com"
port = 12579

print("="*80)
print("TESTING BOTH PASSWORD VARIANTS")
print("="*80)

for name, password in passwords:
    print(f"\nTesting: {name}")
    print(f"Password: {password}")
    
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=0,
            socket_connect_timeout=10,
            decode_responses=False
        )
        
        # Test ping
        response = client.ping()
        print(f"✅ SUCCESS! Ping response: {response}")
        
        # Test set/get
        client.set('test:password:check', f'works_{name}', ex=60)
        value = client.get('test:password:check')
        print(f"✅ Set/Get works! Value: {value}")
        
        # Cleanup
        client.delete('test:password:check')
        
        print(f"\n🎉 CORRECT PASSWORD: {password}")
        print(f"   Use this in your REDIS_URL")
        
        # Show correct URL
        correct_url = f"redis://:{password}@{host}:{port}/0"
        print(f"\n   REDIS_URL={correct_url}")
        
        break
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        continue

print("\n" + "="*80)
