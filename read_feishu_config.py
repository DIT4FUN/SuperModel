import json
with open('/home/treeman/.openclaw/openclaw.json') as f:
    data = json.load(f)
feishu = data.get('channels', {}).get('feishu', {})
print('channels.feishu:', json.dumps(feishu, indent=2)[:800])
plugins = data.get('plugins', {}).get('entries', {}).get('feishu', {})
print('plugins.entries.feishu:', json.dumps(plugins, indent=2)[:800])
