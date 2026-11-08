# dm/android unit tests that need no device: make -C dm/android/tests (clang or gcc; build output in build/, not committed)
B = build
CC ?= cc
.DEFAULT_GOAL := run
run: $(B)/t_movie_zc $(B)/t_ubo_extent $(B)/t_cpu_storage
	$(B)/t_movie_zc
	$(B)/t_ubo_extent
	$(B)/t_cpu_storage
$(B)/t_movie_zc: t_movie_zc.c ../movie_zc.c ../movie_zc.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -o $@ t_movie_zc.c ../movie_zc.c -lpthread
$(B)/t_ubo_extent: t_ubo_extent.c ../gfx_vk_ubo_extent.c ../gfx_vk_ubo_extent.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -o $@ t_ubo_extent.c ../gfx_vk_ubo_extent.c
$(B)/t_cpu_storage: t_cpu_storage.c ../gfx_vk_cpu_storage.c ../gfx_vk_cpu_storage.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -DVCS_TEST -o $@ t_cpu_storage.c ../gfx_vk_cpu_storage.c -lpthread
$(B):
	mkdir -p $(B)
clean:
	rm -rf $(B)
.PHONY: run clean
