#include "trm_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TRM_Memory_State
{
	size_t allocationCount;
};

static struct TRM_Memory_State* pState = NULL;

void TRM_Memory_start()
{
	if (pState != NULL)
	{
		exit(EXIT_FAILURE);
	}

	pState = (struct TRM_Memory_State*)malloc(sizeof(struct TRM_Memory_State));
	if (pState == NULL)
	{
		exit(EXIT_FAILURE);
	}

	memset(pState, 0, sizeof(struct TRM_Memory_State));
}

void TRM_Memory_terminate()
{
	if (pState != NULL)
	{
		if (pState->allocationCount > 0)
		{
			printf("memory leak detected !");
		}
		else
		{
			printf("no memory leak detected ;)");
		}

		free(pState);
		pState = NULL;
	}
}

void TRM_Memory_allocate(size_t size, void** ppMemory)
{
	void* pMemory = malloc(size);
	if (pMemory == NULL)
	{
		exit(EXIT_FAILURE);
	}

	pState->allocationCount += 1;
	*ppMemory = pMemory;
}

void TRM_Memory_deallocate(void* pMemory)
{
	free(pMemory);
	pState->allocationCount -= 1;
}

void TRM_Memory_memzero(size_t size, void* pMemory)
{
	if (pMemory == NULL)
	{
		exit(EXIT_FAILURE);
	}

	memset(pMemory, 0, size);
}

void TRM_Memory_memcpy(size_t size, const void* pSrc, void* pDst)
{
	memcpy(pDst, pSrc, size);
}

void TRM_Memory_realloc(size_t size, void** ppDst)
{
	void* pNewMemory = realloc(*ppDst, size);
	if (pNewMemory == NULL)
	{
		exit(EXIT_FAILURE);
	}

	*ppDst = pNewMemory;
}