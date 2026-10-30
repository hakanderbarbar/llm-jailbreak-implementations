"""JSON-backed, parameterized prompt loading (no evaluation prompts embedded here)."""
import json
from pathlib import Path
from string import Formatter
from typing import Mapping


class PromptLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError('Prompt JSON must map names to string templates')
        self.templates = data

    def render(self, template_name: str, **parameters: object) -> str:
        template = self.templates[template_name]
        required = {field.split('.')[0].split('[')[0] for _, field, _, _ in Formatter().parse(template) if field}
        missing = required - parameters.keys()
        if missing:
            raise ValueError(f'Missing template parameters: {sorted(missing)}')
        return template.format_map(parameters)
