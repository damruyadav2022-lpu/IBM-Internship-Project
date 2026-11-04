from pathlib import Path

env_file = Path('.env')
raw = env_file.read_bytes()
content = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')

NEW_VALUES = {
    'LLM_PROVIDER':    'openai',
    'LLM_API_KEY':     'your_api_key_here',
    'LLM_MODEL':       'gpt-3.5-turbo',
    'LLM_TEMPERATURE': '0.3',
    'LLM_MAX_TOKENS':  '1024',
    'LLM_BASE_URL':    '',
}

lines = content.splitlines()
found_keys = set()
new_lines = []

for line in lines:
    stripped = line.strip()
    matched = False
    for key in NEW_VALUES:
        if stripped.startswith(key + '=') or stripped.startswith('#' + key):
            new_lines.append(key + '=' + NEW_VALUES[key])
            found_keys.add(key)
            matched = True
            break
    if not matched:
        new_lines.append(line)

# Append any keys that were not already in the file
for key, val in NEW_VALUES.items():
    if key not in found_keys:
        new_lines.append(key + '=' + val)
        print('Added: ' + key)

env_file.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
print('Done. .env updated with LLM settings.')
