CPPFLAGS = -std=c++17
LIBS = -lstdc++fs \
	-lopencv_core \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lgdal \
	-lEGL \
	-lGL

all: facadeSynth

facadeSynth: main.o satellite.o
	g++ -o $@ $^ $(LIBS)
	cp $@ ../

main.o: main.cpp
	g++ $(CPPFLAGS) -o $@ -c $<
satellite.o: satellite.cpp
	g++ $(CPPFLAGS) -o $@ -c $<

clean:
	rm -f *.o facadeSynth
