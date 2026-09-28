import json
from collections import Counter

# Load the configuration file
with open('Preprocessing/Config/cell_types_config_collective.json', 'r') as f:
    config = json.load(f)

# Extract all cell types
cell_types = []
for cell_id, cell_info in config['cell_types'].items():
    cell_types.append(cell_info['cell_type'])

# Count unique cell types
unique_cell_types = Counter(cell_types)

print(f"Total number of unique cell types: {len(unique_cell_types)}")
print("\nCell type counts:")
print("-" * 40)

# Sort by count (descending) and display
for cell_type, count in sorted(unique_cell_types.items(), key=lambda x: x[1], reverse=True):
    print(f"{cell_type}: {count}")

print(f"\nTotal samples: {sum(unique_cell_types.values())}") 