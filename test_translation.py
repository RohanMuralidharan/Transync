import json
import sys
import io

# Force UTF-8 output to file
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from hf_inference import translate_onemt, translate_batch

results = []

# Test 1: Telugu → English
r1 = translate_onemt("నమస్కారం, మీరు ఎలా ఉన్నారు?", "tel", "eng")
results.append(("tel→eng", r1))
print(f"tel→eng: {r1}")

# Test 2: English → Hindi
r2 = translate_onemt("Hello, how are you?", "eng", "hin")
results.append(("eng→hin", r2))
print(f"eng→hin: {r2}")

# Test 3: English → Telugu
r3 = translate_onemt("Hello, how are you?", "eng", "tel")
results.append(("eng→tel", r3))
print(f"eng→tel: {r3}")

# Test 4: Code-mixed Hindi → Telugu  
r4 = translate_onemt("मुझे meeting attend करनी है।", "hin", "tel")
results.append(("hin→tel", r4))
print(f"hin→tel: {r4}")

# Test 5: Tamil → Hindi
r5 = translate_onemt("இன்று வானிலை நன்றாக உள்ளது.", "tam", "hin")
results.append(("tam→hin", r5))
print(f"tam→hin: {r5}")

# Save results to JSON
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nAll tests completed. Results saved to test_results.json")