# dm/android unit tests that need no device: make -C dm/android/tests (clang or gcc; build output in build/, not committed)
B = build
CC ?= cc
.DEFAULT_GOAL := run
run: $(B)/t_movie_zc
	$(B)/t_movie_zc
$(B)/t_movie_zc: t_movie_zc.c ../movie_zc.c ../movie_zc.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -o $@ t_movie_zc.c ../movie_zc.c -lpthread
$(B):
	mkdir -p $(B)
clean:
	rm -rf $(B)
.PHONY: run clean
