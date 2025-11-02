from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-mpc-meta-learning",
    version="1.0.0",
    author="WU JIAHAO",
    author_email="u3661739@connect.hku.hk",
    description="RL-Enhanced Model Predictive Control with Meta-Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-meta-pid=training.train_meta_pid:main",
            "train-meta-rl=training.train_meta_rl_combined:main",
            "evaluate-model=evaluation.evaluate_meta_rl:main",
        ],
    },
)

