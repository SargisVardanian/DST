
import pickle
import sys
from pathlib import Path

# Adjust path to find Common_code modules if needed (though we just need to load the dict)
pkl_path = Path("Common_code/pkl_rules/ripper_adult_dst.pkl")

try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Keys in pickle: {list(data.keys())}")
    if "rules" in data:
        print(f"Number of rules: {len(data['rules'])}")
    if "num_classes" in data:
        print(f"num_classes: {data['num_classes']}")
    else:
        print("num_classes key MISSING")
        
except Exception as e:
    print(f"Error loading pickle: {e}")
