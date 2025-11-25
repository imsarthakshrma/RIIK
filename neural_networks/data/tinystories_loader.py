"""
TinyStories Dataset Loader for KOLOSIS
Supports both JSON and TOON formats.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .toon_parser import TOONParser

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories"""
    
    def __init__(self, data_path: str, tokenizer, block_size: int = 128, max_samples: int = None):
        """
        Args:
            data_path: Path to data file (JSON or TOON)
            tokenizer: Character or BPE tokenizer
            block_size: Maximum sequence length
            max_samples: Limit number of samples (for testing)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stories = []
        
        # Load data
        if data_path.endswith('.toon'):
            self.stories = self._load_toon(data_path)
        elif data_path.endswith('.json'):
            self.stories = self._load_json(data_path)
        else:
            # Assume plain text
            self.stories = self._load_text(data_path)
        
        if max_samples:
            self.stories = self.stories[:max_samples]
            
        # Tokenize all stories
        self.encoded_stories = []
        for story in self.stories:
            encoded = tokenizer.encode(story)
            self.encoded_stories.append(encoded)
            
        print(f"Loaded {len(self.stories)} stories")
        
    def _load_json(self, path: str) -> List[str]:
        """Load from JSON format"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of stories
            return [item['story'] if isinstance(item, dict) else item for item in data]
        elif isinstance(data, dict) and 'stories' in data:
            return data['stories']
        else:
            raise ValueError("Unknown JSON format")
    
    def _load_toon(self, path: str) -> List[str]:
        """Load from TOON format"""
        parser = TOONParser()
        with open(path, 'r') as f:
            toon_data = f.read()
        
        data = parser.parse(toon_data)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'stories' in data:
            return data['stories']
        else:
            raise ValueError("Unknown TOON format")
    
    def _load_text(self, path: str) -> List[str]:
        """Load from plain text (one story per line or separated by blank lines)"""
        with open(path, 'r') as f:
            content = f.read()
        
        # Try splitting by double newline first
        stories = content.split('\n\n')
        if len(stories) == 1:
            # Try single newline
            stories = content.split('\n')
        
        # Filter empty stories
        stories = [s.strip() for s in stories if s.strip()]
        return stories
    
    def __len__(self):
        return len(self.encoded_stories)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        encoded = self.encoded_stories[idx]
        
        # Truncate or pad to block_size
        if len(encoded) > self.block_size + 1:
            # Random crop
            start = torch.randint(0, len(encoded) - self.block_size - 1, (1,)).item()
            encoded = encoded[start:start + self.block_size + 1]
        elif len(encoded) < self.block_size + 1:
            # Pad with zeros (or special padding token)
            encoded = encoded + [0] * (self.block_size + 1 - len(encoded))
        
        # Split into input and target
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        
        return x, y


def download_tinystories(output_dir: str = 'data/tinystories'):
    """
    Download TinyStories dataset.
    For now, creates a sample dataset. In production, would download from HuggingFace.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample stories for testing
    sample_stories = [
        "Once upon a time, there was a little girl named Lily. She loved to play in the park.",
        "One day, a big dog came to the park. Lily was scared at first, but the dog was friendly.",
        "Lily and the dog became best friends. They played together every day.",
        "The end.",
    ]
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'sample.json')
    with open(json_path, 'w') as f:
        json.dump({'stories': sample_stories}, f, indent=2)
    
    # Save as TOON (more token-efficient)
    parser = TOONParser()
    toon_str = parser.to_toon({'stories': sample_stories})
    toon_path = os.path.join(output_dir, 'sample.toon')
    with open(toon_path, 'w') as f:
        f.write(toon_str)
    
    print(f"Sample dataset created in {output_dir}")
    print(f"JSON: {json_path}")
    print(f"TOON: {toon_path}")
    
    # Compare file sizes
    json_size = os.path.getsize(json_path)
    toon_size = os.path.getsize(toon_path)
    print(f"\nFile size comparison:")
    print(f"JSON: {json_size} bytes")
    print(f"TOON: {toon_size} bytes")
    print(f"TOON is {(1 - toon_size/json_size)*100:.1f}% smaller")
    
    return json_path, toon_path


if __name__ == "__main__":
    # Download sample dataset
    download_tinystories()
