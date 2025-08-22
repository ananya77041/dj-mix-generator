#!/usr/bin/env python3
"""
Spotify playlist downloader using spotdl
Handles downloading Spotify playlists to local audio files
"""

import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import urllib.parse


class SpotifyPlaylistDownloader:
    """Downloads Spotify playlists using spotdl and manages local files"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the downloader
        
        Args:
            base_dir: Base directory where playlist folders will be created. 
                     If None, uses current working directory.
        """
        if base_dir is None:
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)
        
        self.download_dir = None  # Will be set after getting playlist name
        self.playlist_name = None
        self.is_temp_dir = False
    
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
    
    def _extract_playlist_name_from_output(self, output_text: str) -> Optional[str]:
        """
        Extract playlist name from spotdl output text
        
        Args:
            output_text: Output text from spotdl command
            
        Returns:
            Playlist name if found, None otherwise
        """
        # Look for playlist name patterns in the output
        patterns = [
            r"Found \d+ songs in (.+) \(Playlist\)",  # "Found 25 songs in Jazzy Disco Funk (Playlist)"
            r"Playlist: (.+)",
            r"Processing playlist: (.+)", 
            r"Found playlist: (.+)",
            r'"list_name":\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _sanitize_directory_name(self, name: str) -> str:
        """
        Sanitize playlist name for use as directory name
        
        Args:
            name: Raw playlist name
            
        Returns:
            Sanitized directory name
        """
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100].rstrip()
        # Ensure it's not empty
        if not sanitized:
            sanitized = "spotify_playlist"
        
        return sanitized
    
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
        print(f"🎵 Downloading Spotify playlist: {normalized_url}")
        
        try:
            # Use spotdl's {list-name} variable to automatically create playlist directory
            # This avoids the need for metadata extraction and is more robust
            output_template = "{list-name}/{artists} - {title}.{output-ext}"
            output_path = str(self.base_dir / output_template)
            
            # Prepare spotdl download command with multiple audio sources and highest quality
            cmd = [
                "spotdl",
                "download", 
                normalized_url,
                "--output", output_path,
                "--format", format,
                "--bitrate", "auto",    # Use highest available bitrate
                "--threads", "2",       # Moderate threading for stability
                "--overwrite", "skip",  # Skip existing files
                "--audio", "youtube-music", "youtube", "soundcloud",  # Multiple fallback sources
                "--config"              # Use the config file we modified
            ]
            
            print("⬇️  Running spotdl download with multiple audio sources...")
            print("🎵 Audio sources: YouTube Music → YouTube → SoundCloud")
            print(f"🎧 Format: {format.upper()} (highest quality available)")
            print("📥 Download progress:")
            print("-" * 80)
            
            # Run spotdl command with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream output in real-time and collect it for analysis
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # Print the line immediately for real-time feedback
                    print(line.rstrip())
                    output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Combine all output for analysis
            full_output = ''.join(output_lines)
            
            print("-" * 80)
            
            if return_code != 0:
                raise ValueError(f"spotdl download failed with return code {return_code}")
            
            # Use the collected output for analysis instead of result.stdout
            result_stdout = full_output
            
            # Try to extract playlist name from the output for better user feedback
            playlist_name = self._extract_playlist_name_from_output(result_stdout)
            if not playlist_name:
                playlist_name = "Unknown Playlist"
            
            self.playlist_name = playlist_name
            
            # Count how many songs were found vs downloaded
            found_count = 0
            download_count = 0
            lookup_errors = 0
            
            lines = result_stdout.split('\n')
            for line in lines:
                if 'Found' in line and 'songs in' in line:
                    try:
                        found_count = int(line.split()[1])
                    except:
                        pass
                elif 'Downloaded' in line:
                    download_count += 1
                elif 'LookupError: No results found' in line:
                    lookup_errors += 1
            
            print(f"📊 Found {found_count} songs in playlist, {lookup_errors} failed to find on YouTube/SoundCloud")
            
            # Find the created directory - look for directories that were just created
            potential_dirs = []
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    # Check if this directory has audio files
                    audio_files = self._get_downloaded_files_in_dir(item, format)
                    if audio_files:
                        potential_dirs.append((item, audio_files))
            
            if not potential_dirs:
                if lookup_errors > 0 and download_count == 0:
                    raise ValueError(
                        f"No tracks were downloaded from playlist '{playlist_name}'. "
                        f"All {lookup_errors} tracks failed to be found on YouTube Music, YouTube, or SoundCloud. "
                        f"This playlist contains tracks that are not available on any of these platforms or are too obscure. "
                        f"Try a different playlist with more mainstream tracks."
                    )
                else:
                    raise ValueError(f"No playlist directory with audio files was created for playlist '{playlist_name}'")
            
            # Use the directory with the most audio files (in case multiple exist)
            self.download_dir, downloaded_files = max(potential_dirs, key=lambda x: len(x[1]))
            
            print(f"✅ Successfully downloaded {len(downloaded_files)} tracks to: {self.download_dir}")
            return downloaded_files
            
        except FileNotFoundError:
            raise ValueError(
                "spotdl is not installed. Install it with: pip install spotdl"
            )
        except Exception as e:
            raise ValueError(f"Failed to download playlist: {str(e)}")
    
    def _get_downloaded_files_in_dir(self, directory: Path, format: str) -> List[str]:
        """
        Get list of downloaded files in a specific directory
        
        Args:
            directory: Directory to search in
            format: Audio format to look for
            
        Returns:
            List of file paths in playlist order
        """
        # Find all audio files in the specified directory
        audio_extensions = {format.lower(), "mp3", "wav", "flac", "m4a"}
        audio_files = []
        
        if not directory.exists() or not directory.is_dir():
            return audio_files
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower().lstrip('.') in audio_extensions:
                audio_files.append(str(file_path.absolute()))
        
        # Sort files by name to maintain playlist order
        # spotdl typically downloads with consistent naming
        audio_files.sort()
        
        return audio_files
    
    def _get_downloaded_files(self, format: str) -> List[str]:
        """
        Get list of downloaded files in the correct order
        
        Args:
            format: Audio format to look for
            
        Returns:
            List of file paths in playlist order
        """
        if not self.download_dir:
            return []
        
        return self._get_downloaded_files_in_dir(self.download_dir, format)
    
    def cleanup(self):
        """Clean up temporary download directory (only if it was a temp dir)"""
        import shutil
        if self.is_temp_dir and self.download_dir and self.download_dir.exists():
            print(f"Cleaning up temporary directory: {self.download_dir}")
            shutil.rmtree(self.download_dir)
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            # Only cleanup if it's a temp directory
            if self.is_temp_dir:
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