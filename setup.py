from setuptools import setup, find_packages

setup(
    name="fraud_model",
    version="0.1.0",
    packages=find_packages(),
     install_requires=[
        "pandas>=2.2,<3",
        "numpy>=2.0,<3",
        "scikit-learn>=1.7,<2",
        "xgboost>=3.0,<4",
        "joblib>=1.5,<2",
        "fastapi>=0.100,<1",
        "uvicorn>=0.23,<1",
    ],
    python_requires=">=3.11",
    zip_safe=False,
    include_package_data=True,
    author="Mba DAVID Emeka",
    description="A machine learning package for fraud detection."
)