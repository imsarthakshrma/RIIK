"""
TOON (Token-Oriented Object Notation) Parser for KOLOSIS
Token-efficient data format optimized for LLM training.
"""
import re
from typing import Dict, List, Any, Union

class TOONParser:
    """Parser for TOON format"""
    
    def __init__(self):
        self.indent_size = 2
        
    def parse(self, toon_str: str) -> Union[Dict, List, str, int, float, bool, None]:
        """Parse TOON string to Python object"""
        lines = toon_str.strip().split('\n')
        return self._parse_lines(lines, 0)[0]
    
    def _parse_lines(self, lines: List[str], start_idx: int, base_indent: int = 0) -> tuple:
        """Recursively parse lines"""
        if start_idx >= len(lines):
            return None, start_idx
            
        line = lines[start_idx]
        indent = self._get_indent(line)
        content = line.strip()
        
        # Skip comments and empty lines
        if not content or content.startswith('#'):
            return self._parse_lines(lines, start_idx + 1, base_indent)
        
        # Check for key-value pair
        if ':' in content and not content.startswith('['):
            key, value = content.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Check if value is on next line (object or array)
            if not value:
                # Look ahead for nested content
                if start_idx + 1 < len(lines):
                    next_indent = self._get_indent(lines[start_idx + 1])
                    if next_indent > indent:
                        # Nested object
                        obj, next_idx = self._parse_object(lines, start_idx + 1, next_indent)
                        return {key: obj}, next_idx
                return {key: None}, start_idx + 1
            else:
                # Inline value
                parsed_value = self._parse_value(value)
                return {key: parsed_value}, start_idx + 1
        
        # Array notation: items[N]
        if content.startswith('[') or re.match(r'\w+\[\d+\]', content):
            return self._parse_array(lines, start_idx, indent)
        
        # Standalone value
        return self._parse_value(content), start_idx + 1
    
    def _parse_object(self, lines: List[str], start_idx: int, base_indent: int) -> tuple:
        """Parse object (nested key-value pairs)"""
        obj = {}
        idx = start_idx
        
        while idx < len(lines):
            line = lines[idx]
            indent = self._get_indent(line)
            
            if indent < base_indent:
                break
                
            if indent == base_indent:
                parsed, next_idx = self._parse_lines(lines, idx, base_indent)
                if isinstance(parsed, dict):
                    obj.update(parsed)
                idx = next_idx
            else:
                idx += 1
                
        return obj, idx
    
    def _parse_array(self, lines: List[str], start_idx: int, base_indent: int) -> tuple:
        """Parse array"""
        line = lines[start_idx].strip()
        
        # Extract array size if present
        match = re.match(r'(\w+)?\[(\d+)\]', line)
        if match:
            array_name = match.group(1)
            array_size = int(match.group(2))
            
            # Parse array items
            items = []
            idx = start_idx + 1
            
            while idx < len(lines) and len(items) < array_size:
                item_line = lines[idx]
                item_indent = self._get_indent(item_line)
                
                if item_indent <= base_indent:
                    break
                    
                parsed, next_idx = self._parse_lines(lines, idx, item_indent)
                items.append(parsed)
                idx = next_idx
                
            return items, idx
        
        return [], start_idx + 1
    
    def _parse_value(self, value: str) -> Union[str, int, float, bool, None]:
        """Parse primitive value"""
        value = value.strip()
        
        # Null
        if value == 'null':
            return None
            
        # Boolean
        if value == 'true':
            return True
        if value == 'false':
            return False
            
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        # String (remove quotes if present)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
            
        return value
    
    def _get_indent(self, line: str) -> int:
        """Get indentation level"""
        return len(line) - len(line.lstrip())
    
    def to_toon(self, obj: Any, indent: int = 0) -> str:
        """Convert Python object to TOON string"""
        if obj is None:
            return 'null'
        if isinstance(obj, bool):
            return 'true' if obj else 'false'
        if isinstance(obj, (int, float)):
            return str(obj)
        if isinstance(obj, str):
            return f'"{obj}"'
        if isinstance(obj, list):
            return self._list_to_toon(obj, indent)
        if isinstance(obj, dict):
            return self._dict_to_toon(obj, indent)
        return str(obj)
    
    def _dict_to_toon(self, obj: Dict, indent: int) -> str:
        """Convert dict to TOON"""
        lines = []
        indent_str = ' ' * indent
        
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key}:")
                lines.append(self.to_toon(value, indent + self.indent_size))
            else:
                value_str = self.to_toon(value, 0)
                lines.append(f"{indent_str}{key}: {value_str}")
                
        return '\n'.join(lines)
    
    def _list_to_toon(self, obj: List, indent: int) -> str:
        """Convert list to TOON"""
        lines = []
        indent_str = ' ' * indent
        
        lines.append(f"{indent_str}items[{len(obj)}]")
        for item in obj:
            item_str = self.to_toon(item, indent + self.indent_size)
            lines.append(item_str)
            
        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    parser = TOONParser()
    
    # Test parsing
    toon_text = """
name: "Alice"
age: 30
active: true
scores[3]
  85
  92
  78
"""
    
    result = parser.parse(toon_text)
    print("Parsed:", result)
    
    # Test serialization
    data = {
        "name": "Bob",
        "age": 25,
        "scores": [90, 85, 88]
    }
    toon_str = parser.to_toon(data)
    print("\nTOON format:")
    print(toon_str)
