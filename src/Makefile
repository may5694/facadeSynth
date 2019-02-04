CPPFLAGS = -std=c++17
LIBS = -lEGL \
	-lGL

all: facadeScore

facadeScore: main.o
	g++ -o $@ $^ $(LIBS)
	cp $@ ../

main.o: main.cpp
	g++ $(CPPFLAGS) -o $@ -c $<

clean:
	rm -f *.o facadeScore
