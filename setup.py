#!/usr/bin/env python3
"""
Setup script for DJ Mix Generator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().splitlines()
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        'librosa>=0.9.0',
        'soundfile>=0.10.0', 
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ]

setup(
    name="dj-mix-generator",
    version="2.0.0",
    description="Professional DJ mixing tool with advanced beat matching and transitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DJ Mix Generator Team",
    author_email="",
    url="https://github.com/your-username/dj-mix-generator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        'gui': ['dearpygui>=1.10.0', 'PyQt5', 'tkinter'],
        'dev': ['pytest>=6.0', 'pytest-cov', 'flake8', 'black'],
    },
    entry_points={
        'console_scripts': [
            'dj-mix-generator=cli.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Mixers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    keywords="dj mixing audio beat matching crossfade transitions",
    project_urls={
        "Bug Reports": "https://github.com/your-username/dj-mix-generator/issues",
        "Source": "https://github.com/your-username/dj-mix-generator",
        "Documentation": "https://github.com/your-username/dj-mix-generator/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
)