from pathlib import Path

raw = Path('.env').read_bytes()
content = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')

for line in content.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if any(k in line for k in ['LLM_PROVIDER', 'LLM_MODEL', 'LLM_BASE_URL', 'LLM_API_KEY', 'YOLO_MODEL_PATH']):
        if 'LLM_API_KEY' in line and '=' in line:
            k, v = line.split('=', 1)
            masked = v[:10] + '...' + v[-6:] if len(v) > 16 else v[:4] + '...'
            print(k + '=' + masked + '  (length=' + str(len(v)) + ')')
        else:
            print(line)
