# 9 base tier implemented. Nov9
# Beginning of knowledge_base.py
# Nov6 Cog update implemented 
````python
from collections import deque
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional

# Nov6 Cog internals Start

from internal_process_monitor import InternalProcessMonitor

class TieredKnowledgeBase:
    # Define complexity tiers
    TIERS = {
        'easy': (1, 3),
        'simp': (4, 7),
        'norm': (8, 11),
        'mods': (12, 15),
        'hard': (16, 19),
        'para': (20, 23),
        'vice': (24, 27),
        'zeta': (28, 31),
        'tetris': (32, 35)
    }

    def __init__(self, max_recent_items: int = 100):
        self.knowledge_bases = {tier: {} for tier in self.TIERS.keys()}
        self.recent_updates = deque(maxlen=max_recent_items)
        self.lock = Lock()
        self.knowledge_base_monitor = InternalProcessMonitor()

    def update(self, key: str, value: Any, complexity: int) -> bool:
        tier = self._get_tier(complexity)
        if tier is None:
            return False
        with self.lock:
            if key in self.knowledge_bases[tier]:
                if isinstance(self.knowledge_bases[tier][key], list):
                    if isinstance(value, list):
                        self.knowledge_bases[tier][key].extend(value)
                    else:
                        self.knowledge_bases[tier][key].append(value)
                    self.knowledge_bases[tier][key] = list(set(self.knowledge_bases[tier][key]))
                else:
                    self.knowledge_bases[tier][key] = value
            else:
                self.knowledge_bases[tier][key] = value if not isinstance(value, list) else list(set(value))
            self.recent_updates.append((tier, key, value, complexity))
            self.knowledge_base_monitor.on_knowledge_update(tier, key, value, complexity)
        return True

    def query(self, key: str, complexity_range: Tuple[int, int] = None) -> Dict[str, Any]:
        results = {}
        with self.lock:
            if complexity_range:
                min_comp, max_comp = complexity_range
                relevant_tiers = [tier for tier, (tier_min, tier_max) in self.TIERS.items() if not (tier_max < min_comp or tier_min > max_comp)]
            else:
                relevant_tiers = self.TIERS.keys()
            for tier in relevant_tiers:
                if key in self.knowledge_bases[tier]:
                    results[tier] = self.knowledge_bases[tier][key]
        self.knowledge_base_monitor.on_knowledge_query(key, complexity_range, results)
        return results

    def get_recent_updates(self, n: int = None) -> List[Tuple[str, str, Any, int]]:
        with self.lock:
            updates = list(self.recent_updates)
            if n is not None:
                updates = updates[-n:]
        self.knowledge_base_monitor.on_recent_updates_accessed(updates)
        return updates

    def get_tier_stats(self) -> Dict[str, int]:
        with self.lock:
            tier_stats = {tier: len(kb) for tier, kb in self.knowledge_bases.items()}
        self.knowledge_base_monitor.on_tier_stats_accessed(tier_stats)
        return tier_stats

# Nov6 Cog internals end


    def _get_tier(self, complexity: int) -> Optional[str]:
        """Determine which tier a piece of information belongs to based on its complexity."""
        for tier, (min_comp, max_comp) in self.TIERS.items():
            if min_comp <= complexity <= max_comp:
                return tier
        return None

    def update(self, key: str, value: Any, complexity: int) -> bool:
        """
        Update the knowledge base with new information based on complexity.
        
        Args:
            key: The identifier for the information
            value: The information to store
            complexity: The complexity rating (1-35)
            
        Returns:
            bool: True if update was successful, False if complexity is out of range
        """
        tier = self._get_tier(complexity)
        if tier is None:
            return False

        with self.lock:
            if key in self.knowledge_bases[tier]:
                if isinstance(self.knowledge_bases[tier][key], list):
                    if isinstance(value, list):
                        self.knowledge_bases[tier][key].extend(value)
                    else:
                        self.knowledge_bases[tier][key].append(value)
                    self.knowledge_bases[tier][key] = list(set(self.knowledge_bases[tier][key]))
                else:
                    self.knowledge_bases[tier][key] = value
            else:
                self.knowledge_bases[tier][key] = value if not isinstance(value, list) else list(set(value))
            
            self.recent_updates.append((tier, key, value, complexity))
            return True

    def query(self, key: str, complexity_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Query the knowledge base for information within a specific complexity range.
        
        Args:
            key: The identifier to search for
            complexity_range: Optional tuple of (min_complexity, max_complexity)
            
        Returns:
            Dict containing matches found in relevant tiers
        """
        results = {}
        
        with self.lock:
            if complexity_range:
                min_comp, max_comp = complexity_range
                relevant_tiers = [
                    tier for tier, (tier_min, tier_max) in self.TIERS.items()
                    if not (tier_max < min_comp or tier_min > max_comp)
                ]
            else:
                relevant_tiers = self.TIERS.keys()

            for tier in relevant_tiers:
                if key in self.knowledge_bases[tier]:
                    results[tier] = self.knowledge_bases[tier][key]
                    
        return results

    def get_recent_updates(self, n: int = None) -> List[Tuple[str, str, Any, int]]:
        """
        Get recent updates across all tiers.
        
        Args:
            n: Optional number of recent updates to return
            
        Returns:
            List of tuples (tier, key, value, complexity)
        """
        with self.lock:
            updates = list(self.recent_updates)
            if n is not None:
                updates = updates[-n:]
            return updates

    def get_tier_stats(self) -> Dict[str, int]:
        """
        Get statistics about how many items are stored in each tier.
        
        Returns:
            Dict mapping tier names to item counts
        """
        with self.lock:
            return {
                tier: len(kb) 
                for tier, kb in self.knowledge_bases.items()
            }
            
            ````
# End of knowledge_base.py
