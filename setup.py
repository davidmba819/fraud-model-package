from setuptools import setup, find_packages

setup(
    name="fraud_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas==3.0.3',
        'numpy==2.4.6',
        'scikit-learn==1.9.0',
        'xgboost==3.2.0',
        'joblib==1.5.3'
    ],
    python_requires=">=3.11",
    zip_safe=False,
    include_package_data=True,
    author="Mba DAVID Emeka",
    description="A machine learning package for fraud detection."
)