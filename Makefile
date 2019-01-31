
default: ./build/mnasnet-test
	./build/mnasnet-test

./build/mnasnet-test: ./src/test.cc ./src/mnasnet.hpp
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=/Users/fuchen/opt/libtorch ..
	$(MAKE) -C ./build

clean:
	rm -rf ./build
