#include "trm_containers.h"
#include "trm_memory.h"

#include <stdlib.h>

static void TRM_DynamicArray_grow(struct TRM_DynamicArray* pDynamicArray)
{
	pDynamicArray->elementCapacity *= 2;
	TRM_Memory_realloc(pDynamicArray->elementCapacity * pDynamicArray->elementSize, &pDynamicArray->pData);
}

void TRM_DynamicArray_create(size_t elementSize, struct TRM_DynamicArray* pDynamicArray)
{
	TRM_Memory_allocate(elementSize * 2, &pDynamicArray->pData);

	pDynamicArray->elementSize = elementSize;
	pDynamicArray->elementCount = 0;
	pDynamicArray->elementCapacity = 2;
}

void TRM_DynamicArray_destroy(struct TRM_DynamicArray* pDynamicArray)
{
	TRM_Memory_deallocate(pDynamicArray->pData);
}

void TRM_DynamicArray_push(const void* pElement, struct TRM_DynamicArray* pDynamicArray)
{
	if(pDynamicArray->elementCount == pDynamicArray->elementCapacity)
	{
		TRM_DynamicArray_grow(pDynamicArray);
	}

	TRM_Memory_memcpy(pDynamicArray->elementSize, pElement, (uint8_t*)pDynamicArray->pData + pDynamicArray->elementCount * pDynamicArray->elementSize);
	pDynamicArray->elementCount += 1;
}

void TRM_DynamicArray_at(uint32_t index, const struct TRM_DynamicArray* pDynamicArray, void* pElement)
{
	if(index >= pDynamicArray->elementCount)
	{
		exit(EXIT_FAILURE);
	}

	TRM_Memory_memcpy(pDynamicArray->elementSize, (uint8_t*)pDynamicArray->pData + pDynamicArray->elementSize * index, pElement);
}