"""
Fix Redis import to be optional (graceful degradation)
"""

def fix_redis_import():
    """Make Redis import optional in data_engine.py"""
    
    with open('technic_v4/data_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace hard import with try/except
    old_import = """from technic_v4.infra.logging import get_logger
import redis
from redis.exceptions import RedisError"""
    
    new_import = """from technic_v4.infra.logging import get_logger

# Optional Redis import (graceful degradation)
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisError = Exception  # Fallback"""
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✓ Fixed Redis import to be optional")
    else:
        print("⚠ Redis import pattern not found (may already be fixed)")
    
    # Update _init_redis to check availability
    old_init = """def _init_redis():
    \"\"\"Initialize Redis client (best-effort)\"\"\"
    global _REDIS_CLIENT, _REDIS_ENABLED
    try:"""
    
    new_init = """def _init_redis():
    \"\"\"Initialize Redis client (best-effort)\"\"\"
    global _REDIS_CLIENT, _REDIS_ENABLED
    
    if not REDIS_AVAILABLE:
        _REDIS_ENABLED = False
        logger.info("[REDIS] Redis module not installed, using L1/L2 cache only")
        return
    
    try:"""
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✓ Updated _init_redis() to check availability")
    
    # Write back
    with open('technic_v4/data_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Redis is now optional - code will work without it")

if __name__ == "__main__":
    try:
        fix_redis_import()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
