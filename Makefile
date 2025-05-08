.PHONY: clean build

clean:
	rm -rf build/ dist/ *.egg-info/

build: clean
	python -m build --outdir dist/ 