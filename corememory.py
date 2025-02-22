import json
import time
import logging
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

class MemoryEntry(TypedDict):
    data: Any
    timestamp: float
    access_count: int

class EnhancedHippocampus:
    """Dual memory system with automatic consolidation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.short_term = deque(maxlen=config.get('short_term_capacity', 100))
        self.long_term: Dict[str, MemoryEntry] = {}
        self.persistence_file = config.get('persistence_file', 'long_term_memory.json')
        self.consolidation_interval = config.get('consolidation_interval', 3600)
        self._load_long_term_memory()
        
    def _load_long_term_memory(self) -> None:
        try:
            with open(self.persistence_file, 'r') as f:
                loaded = json.load(f)
                self.long_term = {k: MemoryEntry(**v) for k, v in loaded.items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Memory load failed: {str(e)}")
            self.long_term = {}

    def _save_long_term_memory(self) -> None:
        with open(self.persistence_file, 'w') as f:
            json.dump(self.long_term, f, indent=2)

    def store_memory(self, key: Union[str, float], data: Any, long_term: bool = False) -> None:
        """Store memory in short-term or long-term storage"""
        if long_term:
            self.long_term[str(key)] = MemoryEntry(
                data=data,
                timestamp=time.time(),
                access_count=0
            )
            self._save_long_term_memory()
        else:
            self.short_term.append((key, data, time.time()))

    def recall_memory(self, key: Union[str, float]) -> Optional[Any]:
        """Retrieve memory from storage systems"""
        key_str = str(key)
        for k, data, _ in self.short_term:
            if str(k) == key_str:
                return data
        return self.long_term.get(key_str, {}).get('data')

    def auto_consolidate(self) -> None:
        """Move old memories to long-term storage"""
        cutoff = time.time() - self.consolidation_interval
        for item in list(self.short_term):
            if item[2] < cutoff:
                self.store_memory(item[0], item[1], long_term=True)
                self.short_term.remove(item)

    def get_context(self, query: str, limit: int = 3) -> List[str]:
        """Get relevant context from short-term memory"""
        return [item[1] for item in self.short_term if query in str(item[1])][:limit]