test: ./build/mnasnet-test
	$(MAKE) -C ./build
	./build/mnasnet-test
mnist: ./build/mnist
	./build/mnist

./build/mnist: ./src/mnist.cc ./src/mnasnet.hpp
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=/Users/fuchen/opt/libtorch ..


./build/mnasnet-test: ./src/test.cc ./src/mnasnet.hpp
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=/Users/fuchen/opt/libtorch ..
	$(MAKE) -C ./build

clean:
	rm -rf ./build
