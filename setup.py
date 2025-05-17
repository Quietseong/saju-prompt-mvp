from setuptools import setup, find_packages

setup(
    name="sajumate",
    version="0.1.0",
    description="한국 전통 사주 운세 생성 도구",
    author="SajuMate Team",
    author_email="actual@email.com",
    url="https://github.com/actualusername/sajumate",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "openai>=1.0.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "bitsandbytes>=0.41.0",    # 4bit 양자화에 필요
        "ctransformers>=0.2.0",     # C로 구현된 경량 변환기 (대체 옵션)
        "accelerate>=0.20.0",       # HF 가속
        "sentencepiece>=0.1.99",    # 토큰화
        "huggingface_hub>=0.19.0",  # 허깅페이스 API 접근
        "pytz>=2025.1",             # 시간대 관리
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sajumate=app:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
    keywords="saju, fortune-telling, ai, llm, english",
) 