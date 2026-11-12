# dm/android unit tests that need no device: make -C dm/android/tests (clang or gcc; build output in build/, not committed)
B = build
CC ?= cc
.DEFAULT_GOAL := run
run: $(B)/t_movie_zc $(B)/t_movie_av1 $(B)/t_ubo_extent $(B)/t_cpu_storage $(B)/t_pool_fileback
	$(B)/t_movie_zc
	$(B)/t_movie_av1
	$(B)/t_ubo_extent
	$(B)/t_cpu_storage
	$(B)/t_pool_fileback
$(B)/t_movie_zc: t_movie_zc.c ../movie_zc.c ../movie_zc.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -o $@ t_movie_zc.c ../movie_zc.c -lpthread
$(B)/t_movie_av1: t_movie_av1.c ../movie_av1.c ../movie_av1.h ../../host/hle_movie_av1.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -I../../host -o $@ t_movie_av1.c ../movie_av1.c
$(B)/t_ubo_extent: t_ubo_extent.c ../gfx_vk_ubo_extent.c ../gfx_vk_ubo_extent.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -o $@ t_ubo_extent.c ../gfx_vk_ubo_extent.c
$(B)/t_cpu_storage: t_cpu_storage.c ../gfx_vk_cpu_storage.c ../gfx_vk_cpu_storage.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -DVCS_TEST -o $@ t_cpu_storage.c ../gfx_vk_cpu_storage.c -lpthread
# (Linux only: fallocate, mremap and /proc/self/pagemap; elsewhere the test says it is skipped)
$(B)/t_pool_fileback: t_pool_fileback.c ../gfx_vk_fileback.c ../gfx_vk_fileback.h | $(B)
	$(CC) -O1 -g -Wall -Werror -fsanitize=address,undefined -D_GNU_SOURCE -DFB_TEST -o $@ t_pool_fileback.c $(if $(filter Linux,$(shell uname -s)),../gfx_vk_fileback.c) -lpthread
$(B):
	mkdir -p $(B)
clean:
	rm -rf $(B)
.PHONY: run clean
