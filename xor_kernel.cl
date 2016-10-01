__kernel void vadd(__global unsigned char *data, __global unsigned char *key, const unsigned int key_length, const unsigned int data_length) {
	int i = get_global_id(0);
	if (i < data_length)
		data[i] ^= key[i % key_length];
};
