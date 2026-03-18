#ifndef TRM_MEMORY
#define TRM_MEMORY

#include <stdint.h>

void TRM_Memory_start();

void TRM_Memory_terminate();

void TRM_Memory_allocate(size_t size, void** ppMemory);

void TRM_Memory_deallocate(void* pMemory);

void TRM_Memory_memzero(size_t size, void* pMemory);

void TRM_Memory_memcpy(size_t size, const void* pSrc, void* pDst);

void TRM_Memory_realloc(size_t size, void** ppDst);

#endif