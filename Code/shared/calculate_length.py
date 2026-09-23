import json
from pathlib import Path

# Folder containing your JSON files
folder = Path("./Code/results/crescendo/deepseek-deepseek-v3.2")

total_response_length = 0
total_response_count = 0


def find_responses(obj):
    """Recursively find all values belonging to a 'response' key."""
    responses = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "response" and isinstance(value, str):
                responses.append(value)
            else:
                responses.extend(find_responses(value))

    elif isinstance(obj, list):
        for item in obj:
            responses.extend(find_responses(item))

    return responses


for file_path in folder.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = find_responses(data)

    file_length = sum(len(response) for response in responses)
    file_count = len(responses)

    total_response_length += file_length
    total_response_count += file_count

    print(
        f"{file_path.name}: "
        f"{file_count} responses, "
        f"{file_length} total characters"
    )


# Calculate overall average
if total_response_count > 0:
    average_length = total_response_length / total_response_count
else:
    average_length = 0


print("\n--- Overall results ---")
print(f"Total responses: {total_response_count}")
print(f"Total response length: {total_response_length} characters")
print(f"Average response length: {average_length:.2f} characters")