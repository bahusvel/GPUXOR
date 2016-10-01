#pragma once

#include <stdio.h>
#include <stdlib.h>

char *read_kernel(char *path) {
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		perror("open");
		return NULL;
	}
	fseek(file, 0L, SEEK_END);
	size_t sz = ftell(file);
	rewind(file);
	void *buffer = malloc(sz);
	if (buffer == NULL) {
		perror("malloc");
		goto close_and_exit;
	}
	if (fread(buffer, 1, sz, file) != sz) {
		perror("read");
		goto free_and_exit;
	}
	((char *)buffer)[sz] = 0;
	return buffer;
free_and_exit:
	free(buffer);
close_and_exit:
	fclose(file);
	return NULL;
}
