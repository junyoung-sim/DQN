
output: main.o node.o layer.o neural_network.o dqn.o
	g++ -std=c++20 main.o node.o layer.o neural_network.o dqn.o -o exec
	rm *.o

main.o: ./src/main.cpp
	g++ -std=c++20 -c ./src/main.cpp

node.o: ./src/node.cpp
	g++ -std=c++20 -c ./src/node.cpp

layer.o: ./src/layer.cpp
	g++ -std=c++20 -c ./src/layer.cpp

neural_network.o: ./src/neural_network.cpp
	g++ -std=c++20 -c ./src/neural_network.cpp

dqn.o: ./src/dqn.cpp
	g++ -std=c++20 -c ./src/dqn.cpp
