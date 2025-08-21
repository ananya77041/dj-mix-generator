#!/usr/bin/env python3
"""
Key matching utilities for harmonic mixing
"""

from typing import List, Tuple, Dict
try:
    from ..core.models import Track
except ImportError:
    # Fallback for direct execution - use full path to avoid conflicts
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from core.models import Track


class KeyMatcher:
    """Handles key compatibility and harmonic mixing logic"""
    
    def __init__(self):
        # Circle of Fifths mapping for key relationships
        self.key_mapping = {
            # Major keys
            'C major': {'perfect': ['C major', 'A minor'], 'compatible': ['G major', 'F major', 'E minor', 'D minor'], 'semitone': ['C# major', 'B major']},
            'C# major': {'perfect': ['C# major', 'A# minor'], 'compatible': ['G# major', 'F# major', 'F minor', 'D# minor'], 'semitone': ['D major', 'C major']},
            'D major': {'perfect': ['D major', 'B minor'], 'compatible': ['A major', 'G major', 'F# minor', 'E minor'], 'semitone': ['D# major', 'C# major']},
            'D# major': {'perfect': ['D# major', 'C minor'], 'compatible': ['A# major', 'G# major', 'G minor', 'F minor'], 'semitone': ['E major', 'D major']},
            'E major': {'perfect': ['E major', 'C# minor'], 'compatible': ['B major', 'A major', 'G# minor', 'F# minor'], 'semitone': ['F major', 'D# major']},
            'F major': {'perfect': ['F major', 'D minor'], 'compatible': ['C major', 'A# major', 'A minor', 'G minor'], 'semitone': ['F# major', 'E major']},
            'F# major': {'perfect': ['F# major', 'D# minor'], 'compatible': ['C# major', 'B major', 'A# minor', 'G# minor'], 'semitone': ['G major', 'F major']},
            'G major': {'perfect': ['G major', 'E minor'], 'compatible': ['D major', 'C major', 'B minor', 'A minor'], 'semitone': ['G# major', 'F# major']},
            'G# major': {'perfect': ['G# major', 'F minor'], 'compatible': ['D# major', 'C# major', 'C minor', 'A# minor'], 'semitone': ['A major', 'G major']},
            'A major': {'perfect': ['A major', 'F# minor'], 'compatible': ['E major', 'D major', 'C# minor', 'B minor'], 'semitone': ['A# major', 'G# major']},
            'A# major': {'perfect': ['A# major', 'G minor'], 'compatible': ['F major', 'D# major', 'D minor', 'C minor'], 'semitone': ['B major', 'A major']},
            'B major': {'perfect': ['B major', 'G# minor'], 'compatible': ['F# major', 'E major', 'D# minor', 'C# minor'], 'semitone': ['C major', 'A# major']},
            
            # Minor keys
            'A minor': {'perfect': ['A minor', 'C major'], 'compatible': ['E minor', 'D minor', 'G major', 'F major'], 'semitone': ['A# minor', 'G# minor']},
            'A# minor': {'perfect': ['A# minor', 'C# major'], 'compatible': ['F minor', 'D# minor', 'G# major', 'F# major'], 'semitone': ['B minor', 'A minor']},
            'B minor': {'perfect': ['B minor', 'D major'], 'compatible': ['F# minor', 'E minor', 'A major', 'G major'], 'semitone': ['C minor', 'A# minor']},
            'C minor': {'perfect': ['C minor', 'D# major'], 'compatible': ['G minor', 'F minor', 'A# major', 'G# major'], 'semitone': ['C# minor', 'B minor']},
            'C# minor': {'perfect': ['C# minor', 'E major'], 'compatible': ['G# minor', 'F# minor', 'B major', 'A major'], 'semitone': ['D minor', 'C minor']},
            'D minor': {'perfect': ['D minor', 'F major'], 'compatible': ['A minor', 'G minor', 'C major', 'A# major'], 'semitone': ['D# minor', 'C# minor']},
            'D# minor': {'perfect': ['D# minor', 'F# major'], 'compatible': ['A# minor', 'G# minor', 'C# major', 'B major'], 'semitone': ['E minor', 'D minor']},
            'E minor': {'perfect': ['E minor', 'G major'], 'compatible': ['B minor', 'A minor', 'D major', 'C major'], 'semitone': ['F minor', 'D# minor']},
            'F minor': {'perfect': ['F minor', 'G# major'], 'compatible': ['C minor', 'A# minor', 'D# major', 'C# major'], 'semitone': ['F# minor', 'E minor']},
            'F# minor': {'perfect': ['F# minor', 'A major'], 'compatible': ['C# minor', 'B minor', 'E major', 'D major'], 'semitone': ['G minor', 'F minor']},
            'G minor': {'perfect': ['G minor', 'A# major'], 'compatible': ['D minor', 'C minor', 'F major', 'D# major'], 'semitone': ['G# minor', 'F# minor']},
            'G# minor': {'perfect': ['G# minor', 'B major'], 'compatible': ['D# minor', 'C# minor', 'F# major', 'E major'], 'semitone': ['A minor', 'G minor']},
        }
        
        # Circle of Fifths ordering for sorting (C major = 0, G major = 1, etc.)
        self.key_sort_order = {
            # Major keys (circle of fifths)
            'C major': 0, 'G major': 1, 'D major': 2, 'A major': 3, 'E major': 4, 'B major': 5,
            'F# major': 6, 'C# major': 7, 'G# major': 8, 'D# major': 9, 'A# major': 10, 'F major': 11,
            
            # Minor keys (relative minors, offset by 20 to separate from majors)
            'A minor': 20, 'E minor': 21, 'B minor': 22, 'F# minor': 23, 'C# minor': 24, 'G# minor': 25,
            'D# minor': 26, 'A# minor': 27, 'F minor': 28, 'C minor': 29, 'G minor': 30, 'D minor': 31,
        }
    
    def _get_key_sort_value(self, key: str) -> int:
        """Get numeric sort value for key based on circle of fifths"""
        return self.key_sort_order.get(key, 999)  # Unknown keys sort to end
    
    def get_compatibility_score(self, key1: str, key2: str) -> int:
        """
        Get compatibility score between two keys
        Returns: 3 = perfect match, 2 = compatible, 1 = semitone, 0 = incompatible
        """
        if key1 not in self.key_mapping or key2 not in self.key_mapping:
            return 0
        
        relationships = self.key_mapping[key1]
        
        if key2 in relationships['perfect']:
            return 3
        elif key2 in relationships['compatible']:
            return 2
        elif key2 in relationships['semitone']:
            return 1
        else:
            return 0
    
    def find_best_next_track(self, current_key: str, remaining_tracks: List[Track]) -> Tuple[Track, int]:
        """
        Find the best matching track from remaining tracks
        Returns tuple of (best_track, best_score)
        """
        if not remaining_tracks:
            raise ValueError("No remaining tracks to choose from")
        
        best_track = remaining_tracks[0]
        best_score = self.get_compatibility_score(current_key, best_track.key)
        
        for track in remaining_tracks[1:]:
            score = self.get_compatibility_score(current_key, track.key)
            if score > best_score:
                best_track = track
                best_score = score
        
        return best_track, best_score
    
    def reorder_tracks_by_key(self, tracks: List[Track]) -> List[Track]:
        """
        Reorder tracks to optimize harmonic flow using greedy algorithm
        Starts with the first track and finds the best match for each subsequent track
        """
        if len(tracks) <= 1:
            return tracks
        
        ordered_tracks = [tracks[0]]  # Start with first track
        remaining_tracks = tracks[1:].copy()
        
        print("Reordering tracks for optimal key matching...")
        print(f"Starting with: {tracks[0].filepath.name} ({tracks[0].key})")
        
        while remaining_tracks:
            current_key = ordered_tracks[-1].key
            best_track, score = self.find_best_next_track(current_key, remaining_tracks)
            
            # Get score description
            score_desc = {3: "perfect", 2: "compatible", 1: "semitone", 0: "clash"}[score]
            
            print(f"  Next: {best_track.filepath.name} ({best_track.key}) - {score_desc} match")
            
            ordered_tracks.append(best_track)
            remaining_tracks.remove(best_track)
        
        print("Track reordering complete!\n")
        return ordered_tracks
    
    def analyze_track_flow(self, tracks: List[Track]) -> Dict:
        """
        Analyze the harmonic flow of a track list
        Returns statistics about key compatibility
        """
        if len(tracks) <= 1:
            return {"transitions": 0, "scores": [], "average_score": 0}
        
        scores = []
        transitions = []
        
        for i in range(len(tracks) - 1):
            current_key = tracks[i].key
            next_key = tracks[i + 1].key
            score = self.get_compatibility_score(current_key, next_key)
            scores.append(score)
            transitions.append({
                "from": current_key,
                "to": next_key,
                "score": score,
                "from_track": tracks[i].filepath.name,
                "to_track": tracks[i + 1].filepath.name
            })
        
        return {
            "transitions": len(transitions),
            "scores": scores,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "details": transitions
        }