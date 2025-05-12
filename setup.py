from setuptools import setup, find_packages

setup(
    name="chat2table",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.2",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
) 