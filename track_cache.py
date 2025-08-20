#!/usr/bin/env python3
"""
Track analysis caching system for DJ Mix Generator
Persists analysis results to avoid re-analyzing the same tracks
"""

import json
import hashlib
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict
from models import Track


class TrackCache:
    """Handles caching of track analysis results"""
    
    def __init__(self, cache_dir: str = None):
        # Default cache directory in user's home
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.dj_mix_generator_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate files for metadata and audio data
        self.metadata_file = self.cache_dir / "track_metadata.json"
        self.audio_data_dir = self.cache_dir / "audio_data"
        self.audio_data_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load track metadata from cache file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load track cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save track metadata to cache file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save track cache metadata: {e}")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of the audio file for unique identification"""
        try:
            hasher = hashlib.sha256()
            
            # Read file in chunks to handle large files
            with open(filepath, 'rb') as f:
                # Read first 64KB, middle 64KB, and last 64KB for speed
                # This should be unique enough for most use cases
                chunk_size = 65536  # 64KB
                
                # First chunk
                chunk = f.read(chunk_size)
                if chunk:
                    hasher.update(chunk)
                
                # Get file size
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                if file_size > chunk_size * 2:
                    # Middle chunk
                    f.seek(file_size // 2)
                    chunk = f.read(chunk_size)
                    if chunk:
                        hasher.update(chunk)
                    
                    # Last chunk
                    f.seek(max(0, file_size - chunk_size))
                    chunk = f.read(chunk_size)
                    if chunk:
                        hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except IOError as e:
            print(f"Warning: Could not calculate hash for {filepath}: {e}")
            return None
    
    def _get_cache_key(self, filepath: str, manual_downbeats: bool = False) -> Optional[str]:
        """
        Generate cache key based on file hash and modification time
        Returns None if file cannot be accessed
        """
        try:
            file_stat = os.stat(filepath)
            file_hash = self._calculate_file_hash(filepath)
            
            if file_hash is None:
                return None
            
            # Include file size, modification time, and downbeat mode for cache isolation
            downbeat_suffix = "_manual" if manual_downbeats else "_auto"
            cache_key = f"{file_hash}_{file_stat.st_size}_{int(file_stat.st_mtime)}{downbeat_suffix}"
            return cache_key
            
        except OSError as e:
            print(f"Warning: Could not access file {filepath}: {e}")
            return None
    
    def _serialize_numpy_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays to lists for JSON serialization"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_numpy_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert lists back to numpy arrays"""
        deserialized = {}
        array_fields = ['beats', 'downbeats']  # Fields that should be numpy arrays
        
        for key, value in data.items():
            if key in array_fields and isinstance(value, list):
                deserialized[key] = np.array(value)
            else:
                deserialized[key] = value
        return deserialized
    
    def is_cached(self, filepath: str, manual_downbeats: bool = False) -> bool:
        """Check if track analysis is already cached"""
        cache_key = self._get_cache_key(filepath, manual_downbeats)
        if cache_key is None:
            return False
        
        return cache_key in self.metadata
    
    def get_cached_analysis(self, filepath: str, manual_downbeats: bool = False) -> Optional[Track]:
        """
        Retrieve cached track analysis
        Returns None if not found or if cache is invalid
        """
        cache_key = self._get_cache_key(filepath, manual_downbeats)
        if cache_key is None or cache_key not in self.metadata:
            return None
        
        try:
            # Get metadata
            cached_data = self.metadata[cache_key].copy()
            
            # Load audio data from separate pickle file
            audio_file = self.audio_data_dir / f"{cache_key}.pkl"
            if not audio_file.exists():
                print(f"Warning: Audio data file missing for cached track")
                return None
            
            with open(audio_file, 'rb') as f:
                audio_data = pickle.load(f)
            
            # Combine metadata with audio data
            cached_data['audio'] = audio_data['audio']
            cached_data['filepath'] = Path(filepath)
            
            # Deserialize numpy arrays
            cached_data = self._deserialize_numpy_arrays(cached_data)
            
            # Create Track object
            track = Track(**cached_data)
            
            return track
            
        except (IOError, pickle.PickleError, KeyError, TypeError) as e:
            print(f"Warning: Could not load cached analysis for {filepath}: {e}")
            # Remove corrupted cache entry
            self._remove_cache_entry(cache_key)
            return None
    
    def cache_analysis(self, track: Track, manual_downbeats: bool = False):
        """Cache track analysis results"""
        cache_key = self._get_cache_key(str(track.filepath), manual_downbeats)
        if cache_key is None:
            return
        
        try:
            # Prepare metadata (everything except audio data)
            track_dict = asdict(track)
            audio_data = {'audio': track_dict.pop('audio')}
            
            # Convert Path to string for JSON serialization
            track_dict['filepath'] = str(track.filepath)
            
            # Serialize numpy arrays in metadata
            track_dict = self._serialize_numpy_arrays(track_dict)
            
            # Save metadata to JSON
            self.metadata[cache_key] = track_dict
            self._save_metadata()
            
            # Save audio data to separate pickle file
            audio_file = self.audio_data_dir / f"{cache_key}.pkl"
            with open(audio_file, 'wb') as f:
                pickle.dump(audio_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  Cached analysis for {track.filepath.name}")
            
        except (IOError, pickle.PickleError) as e:
            print(f"Warning: Could not cache analysis for {track.filepath}: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a corrupted or invalid cache entry"""
        try:
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            
            audio_file = self.audio_data_dir / f"{cache_key}.pkl"
            if audio_file.exists():
                audio_file.unlink()
                
        except (IOError, OSError) as e:
            print(f"Warning: Could not remove cache entry {cache_key}: {e}")
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Remove all audio data files
            for audio_file in self.audio_data_dir.glob("*.pkl"):
                audio_file.unlink()
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            print("Track analysis cache cleared")
            
        except (IOError, OSError) as e:
            print(f"Warning: Could not clear cache completely: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache"""
        try:
            cache_size = 0
            audio_files = list(self.audio_data_dir.glob("*.pkl"))
            
            for audio_file in audio_files:
                cache_size += audio_file.stat().st_size
            
            if self.metadata_file.exists():
                cache_size += self.metadata_file.stat().st_size
            
            return {
                "cached_tracks": len(self.metadata),
                "cache_size_mb": cache_size / (1024 * 1024),
                "cache_directory": str(self.cache_dir)
            }
            
        except OSError:
            return {
                "cached_tracks": len(self.metadata),
                "cache_size_mb": 0,
                "cache_directory": str(self.cache_dir)
            }
    
    def cleanup_orphaned_files(self):
        """Remove audio data files that don't have corresponding metadata entries"""
        try:
            audio_files = set(f.stem for f in self.audio_data_dir.glob("*.pkl"))
            metadata_keys = set(self.metadata.keys())
            
            orphaned = audio_files - metadata_keys
            
            for orphaned_key in orphaned:
                audio_file = self.audio_data_dir / f"{orphaned_key}.pkl"
                audio_file.unlink()
                print(f"Removed orphaned cache file: {orphaned_key}")
            
            if orphaned:
                print(f"Cleaned up {len(orphaned)} orphaned cache files")
            
        except OSError as e:
            print(f"Warning: Could not cleanup orphaned files: {e}")