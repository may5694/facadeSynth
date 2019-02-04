CPPFLAGS = -std=c++17
LIBS = -lstdc++fs \
	-lEGL \
	-lGL

all: facadeSynth

facadeSynth: main.o
	g++ -o $@ $^ $(LIBS)
	cp $@ ../

main.o: main.cpp
	g++ $(CPPFLAGS) -o $@ -c $<

clean:
	rm -f *.o facadeSynth
