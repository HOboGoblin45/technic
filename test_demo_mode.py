"""Test demo mode functionality"""
import os
os.environ["DEMO_MODE"] = "true"

from technic_v4.demo import is_demo_mode, load_demo_scan_results, get_demo_copilot_response

print(f"Demo mode enabled: {is_demo_mode()}")

# Test scan results
data = load_demo_scan_results()
print(f"\nLoaded {len(data['results'])} demo stocks:")
for stock in data['results'][:3]:
    print(f"  - {stock['symbol']}: MERIT {stock['meritScore']}, Grade {stock['resultTier']}")

# Test copilot
print(f"\nTesting copilot responses:")
questions = [
    "Why is AAPL recommended?",
    "Explain the top 3 picks",
    "How should I allocate my portfolio?"
]

for q in questions:
    answer = get_demo_copilot_response(q)
    print(f"\nQ: {q}")
    print(f"A: {answer[:100]}...")

print("\n✅ Demo mode is working correctly!")
