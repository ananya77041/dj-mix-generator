#!/usr/bin/env python3
"""
Spotify playlist downloader using spotdl
Handles downloading Spotify playlists to local audio files
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import urllib.parse


class SpotifyPlaylistDownloader:
    """Downloads Spotify playlists using spotdl and manages local files"""
    
    def __init__(self, download_dir: Optional[str] = None):
        """
        Initialize the downloader
        
        Args:
            download_dir: Directory to download tracks to. If None, uses temp directory.
        """
        if download_dir is None:
            self.download_dir = Path(tempfile.mkdtemp(prefix="dj_mix_spotify_"))
        else:
            self.download_dir = Path(download_dir)
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Spotify tracks will be downloaded to: {self.download_dir}")
    
    def _is_valid_spotify_playlist_url(self, url: str) -> bool:
        """Validate if URL is a valid Spotify playlist URL"""
        spotify_patterns = [
            r'^https://open\.spotify\.com/playlist/[a-zA-Z0-9]+',
            r'^spotify:playlist:[a-zA-Z0-9]+',
        ]
        return any(re.match(pattern, url) for pattern in spotify_patterns)
    
    def _normalize_spotify_url(self, url: str) -> str:
        """Normalize Spotify URL to a standard format"""
        # Remove any query parameters or extra parts
        if '?' in url:
            url = url.split('?')[0]
        
        # Convert spotify: URI to https URL if needed
        if url.startswith('spotify:playlist:'):
            playlist_id = url.split(':')[-1]
            url = f"https://open.spotify.com/playlist/{playlist_id}"
        
        return url
    
    def download_playlist(self, spotify_url: str, format: str = "wav") -> List[str]:
        """
        Download a Spotify playlist to local audio files
        
        Args:
            spotify_url: Spotify playlist URL or URI
            format: Audio format to download (default: wav for best quality)
            
        Returns:
            List of paths to downloaded audio files in playlist order
            
        Raises:
            ValueError: If URL is invalid or download fails
        """
        if not self._is_valid_spotify_playlist_url(spotify_url):
            raise ValueError(f"Invalid Spotify playlist URL: {spotify_url}")
        
        normalized_url = self._normalize_spotify_url(spotify_url)
        print(f"Downloading Spotify playlist: {normalized_url}")
        
        try:
            # Prepare spotdl command
            cmd = [
                "spotdl",
                "download",
                normalized_url,
                "--output", str(self.download_dir),
                "--format", format,
                "--bitrate", "320k",  # High quality
                "--threads", "4",     # Parallel downloads
                "--overwrite", "skip" # Skip existing files
            ]
            
            print("Running spotdl download...")
            print(f"Command: {' '.join(cmd)}")
            
            # Run spotdl command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.download_dir
            )
            
            if result.returncode != 0:
                print(f"spotdl stderr: {result.stderr}")
                raise ValueError(f"spotdl download failed: {result.stderr}")
            
            print(f"spotdl output: {result.stdout}")
            
            # Find downloaded files in order
            downloaded_files = self._get_downloaded_files(format)
            
            if not downloaded_files:
                raise ValueError("No files were downloaded")
            
            print(f"Successfully downloaded {len(downloaded_files)} tracks")
            return downloaded_files
            
        except FileNotFoundError:
            raise ValueError(
                "spotdl is not installed. Install it with: pip install spotdl"
            )
        except Exception as e:
            raise ValueError(f"Failed to download playlist: {str(e)}")
    
    def _get_downloaded_files(self, format: str) -> List[str]:
        """
        Get list of downloaded files in the correct order
        
        Args:
            format: Audio format to look for
            
        Returns:
            List of file paths in playlist order
        """
        # Find all audio files in download directory
        audio_extensions = {format.lower(), "mp3", "wav", "flac", "m4a"}
        audio_files = []
        
        for file_path in self.download_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower().lstrip('.') in audio_extensions:
                audio_files.append(str(file_path.absolute()))
        
        # Sort files by name to maintain playlist order
        # spotdl typically downloads with consistent naming
        audio_files.sort()
        
        return audio_files
    
    def cleanup(self):
        """Clean up temporary download directory"""
        import shutil
        if self.download_dir.exists() and "dj_mix_spotify_" in str(self.download_dir):
            print(f"Cleaning up temporary directory: {self.download_dir}")
            shutil.rmtree(self.download_dir)
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            # Only cleanup if it's a temp directory
            if "dj_mix_spotify_" in str(self.download_dir):
                self.cleanup()
        except:
            pass  # Ignore cleanup errors


def download_spotify_playlist(spotify_url: str, download_dir: Optional[str] = None) -> List[str]:
    """
    Convenience function to download a Spotify playlist
    
    Args:
        spotify_url: Spotify playlist URL or URI
        download_dir: Directory to download to (uses temp if None)
        
    Returns:
        List of downloaded audio file paths in playlist order
    """
    downloader = SpotifyPlaylistDownloader(download_dir)
    return downloader.download_playlist(spotify_url)


if __name__ == "__main__":
    # Test the downloader
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python spotify_downloader.py <spotify_playlist_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    try:
        files = download_spotify_playlist(url)
        print(f"\nDownloaded files:")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)