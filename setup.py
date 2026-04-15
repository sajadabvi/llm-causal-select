from setuptools import setup, find_packages

setup(
    name="llm-causal-select",
    version="0.1.0",
    description=(
        "LoRA fine-tuned LLM for causal DAG selection from RASL equivalence classes. "
        "Extends the gunfolds causal discovery toolkit."
    ),
    author="Sajad Abvi",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "train": [
            "bitsandbytes>=0.43.0",
            "datasets>=2.18.0",
        ],
        "track": [
            "wandb>=0.17.0",
            "mlflow>=2.12.0",
        ],
    },
)
