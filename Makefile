all:
	g++ -O3 -std=c++11 main.cc\
	   	-lstdc++fs\
	   	`pkg-config --cflags --libs tesseract opencv4`\
	   	-o yoyakbot
clean:
	rm yoyakbot
