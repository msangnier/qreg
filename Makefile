PYTHON ?= python
CYTHON ?= cython

CYTHONSRC= $(wildcard qreg/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f qreg/*.c qreg/*.html
	rm -f `find qreg -name "*.pyc"`
	rm -f `find qreg -name "*.so"`
	rm -rf `find qreg -name "*pycache*"`
	rm -rf build
	rm -rf *egg-info
	rm -rf dist

%.cpp: %.pyx
	$(CYTHON) --cplus $<

