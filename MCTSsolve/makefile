CC=gcc
CFLAGS= -Wall -DNLOG -O3 -DNDEBUG
LFLAGS= -DSOLVER -lgmp -lm -L./depqbf -lqdpll -I./depqbf

MCTSsolve: MCTSsolve.o makefile
	$(CC) $(CFLAGS) -o $@ MCTSsolve.o $(LFLAGS)
MCTSsolve.o: MCTSsolve.c makefile
	$(CC) $(CFLAGS) -c MCTSsolve.c $(LFLAGS)

clean:
	rm -f `cat .gitignore`
