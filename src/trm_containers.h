#ifndef TRM_CONTAINERS
#define TRM_CONTAINERS

#include <stdint.h>

struct TRM_DynamicArray
{
	size_t elementSize;
	uint32_t elementCount;
	uint32_t elementCapacity;
	void* pData;
};

void TRM_DynamicArray_create(size_t elementSize, struct TRM_DynamicArray* pDynamicArray);

void TRM_DynamicArray_destroy(struct TRM_DynamicArray* pDynamicArray);

void TRM_DynamicArray_push(const void* pElement, struct TRM_DynamicArray* pDynamicArray);

void TRM_DynamicArray_at(uint32_t index, const struct TRM_DynamicArray* pDynamicArray, void* pElement);

#endif