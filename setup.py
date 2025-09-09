#!/usr/bin/env python3
"""
Setup configuration for RADA - Hybrid RAG System
"""

from setuptools import setup, find_packages

# Leer README para la descripción larga
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "RADA - Hybrid RAG System for Technical Support Optimization"

# Leer requirements.txt para las dependencias
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip()
                        and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "chromadb>=0.4.15",
        "sentence-transformers>=2.2.2",
        "anthropic>=0.32.0",
        "ollama>=0.1.7",
        "beautifulsoup4>=4.12.3",
        "html2text>=2020.1.16",
        "numpy>=1.24.3",
        "requests>=2.31.0"
    ]

setup(
    name="rada-rag-system",
    version="61.1.0",
    author="Andrés García",
    author_email="your.email@example.com",  # Opcional
    description="Hybrid RAG system for technical support optimization in fintech companies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rada-rag-system",
    project_urls={
        "Bug Reports": "https://github.com/your-username/rada-rag-system/issues",
        "Source": "https://github.com/your-username/rada-rag-system",
        "Documentation": "https://github.com/your-username/rada-rag-system/blob/main/README.md"
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-mock>=3.10",
        ]
    },
    entry_points={
        "console_scripts": [
            "rada=rada.app_simple_hybrid:main",  # Opcional: comando 'rada' en terminal
        ],
    },
    include_package_data=True,
    keywords="rag, llm, fintech, technical-support, hybrid-ai, confluence, chromadb",
    license="MIT",
)
