import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Convert each line of a text file into a JSON object and save it to a file.")
parser.add_argument('--input', type=str, required=True, help="Path to the input text file.")
parser.add_argument('--output', type=str, required=True, help="Path to save the output JSON file.")
args = parser.parse_args()

# Read input and process JSON lines
json_objects = []
with open(args.input, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        try:
            json_object = json.loads(line.strip())  # Attempt to parse each line
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {i + 1}: {e}")
            print(f"Problematic line: {line}")
            continue  # Skip lines that fail to parse

# Write JSON objects to output file
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(json_objects, f, indent=4, ensure_ascii=False)

print(f"Conversion complete. Results saved to {args.output}")
