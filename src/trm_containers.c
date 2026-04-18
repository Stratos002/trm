#include "trm_containers.h"
#include "trm_memory.h"

#include <stdlib.h>

/* Dynamic Array */

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

/* Arena */

void TRM_Arena_create(size_t elementSize, uint32_t elementCapacity, struct TRM_Arena* pArena)
{
	TRM_Memory_allocate(elementSize * elementCapacity, &pArena->pData);
	TRM_Memory_allocate(sizeof(uint32_t) * elementCapacity, (void**)&pArena->pFreeIndices);
	pArena->elementSize = elementSize;
	pArena->elementCapacity = elementCapacity;

	for(uint32_t i = 0; i < elementCapacity; ++i)
	{
		pArena->pFreeIndices[i] = elementCapacity - i - 1;
	}
	
	pArena->freeIndexCount = elementCapacity;
}

void TRM_Arena_destroy(struct TRM_Arena* pArena)
{
	TRM_Memory_deallocate(pArena->pData);
	TRM_Memory_deallocate(pArena->pFreeIndices);
}

void TRM_Arena_add(const void* pElement, struct TRM_Arena* pArena, uint32_t* pElementIndex)
{
	uint32_t elementIndex = pArena->pFreeIndices[pArena->freeIndexCount - 1];
	TRM_Memory_memcpy(pArena->elementSize, pElement, (uint8_t*)pArena->pData + pArena->elementSize * elementIndex);
	--pArena->freeIndexCount;
	*pElementIndex = elementIndex;
}

void TRM_Arena_remove(uint32_t elementIndex, struct TRM_Arena* pArena)
{
	pArena->pFreeIndices[pArena->freeIndexCount] = elementIndex;
	++pArena->freeIndexCount;
}

void TRM_Arena_get(uint32_t elementIndex, struct TRM_Arena arena, void** ppElement)
{
	*ppElement = (uint8_t*)arena.pData + arena.elementSize * elementIndex;
}

/* Linked List */

static void TRM_LinkedList_createNode(const void* pElement, const struct TRM_LinkedList* pLinkedList, struct TRM_LinkedList_Node** ppNode)
{
	TRM_Memory_allocate(sizeof(struct TRM_LinkedList_Node), (void**)ppNode);
	(*ppNode)->pNextNode = NULL;
	TRM_Memory_allocate(pLinkedList->elementSize, (void**)&((*ppNode)->pData));
	TRM_Memory_memcpy(pLinkedList->elementSize, pElement, (*ppNode)->pData);
}

void TRM_LinkedList_create(size_t elementSize, struct TRM_LinkedList* pLinkedList)
{
	pLinkedList->elementSize = elementSize;
	pLinkedList->elementCount = 0;
	pLinkedList->pFirstNode = NULL;
}

void TRM_LinkedList_destroy(struct TRM_LinkedList* pDoublyLinkedList)
{
	struct TRM_LinkedList_Node* pNode = pDoublyLinkedList->pFirstNode;
	while(pNode != NULL)
	{
		struct TRM_LinkedList_Node* pNextNode = pNode->pNextNode;
		free(pNode->pData);
		free(pNode);
		pNode = pNextNode;
	}
}

void TRM_LinkedList_push(const void* pElement, struct TRM_LinkedList* pLinkedList)
{
	struct TRM_LinkedList_Node* pNewNode = NULL;
	TRM_LinkedList_createNode(pElement, pLinkedList, &pNewNode);

	if(pLinkedList->pFirstNode == NULL)
	{
		pLinkedList->pFirstNode = pNewNode;
	}
	else
	{
		struct TRM_LinkedList_Node* pNode = pLinkedList->pFirstNode;
		while(pNode->pNextNode != NULL)
		{
			pNode = pNode->pNextNode;
		}

		pNode->pNextNode = pNewNode;
	}

	pLinkedList->elementCount += 1;
}

void TRM_LinkedList_insertAfter(const void* pElement, struct TRM_LinkedList_Node* pNode, struct TRM_LinkedList* pLinkedList)
{
	if(pNode == NULL)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_LinkedList_Node* pNewNode = NULL;
	TRM_LinkedList_createNode(pElement, pLinkedList, &pNewNode);

	pNewNode->pNextNode = pNode->pNextNode;
	pNode->pNextNode = pNewNode;
	pLinkedList->elementCount += 1;
}

void TRM_LinkedList_delete(struct TRM_LinkedList_Node* pNode, struct TRM_LinkedList* pLinkedList)
{
	if(pNode == NULL)
	{
		exit(EXIT_FAILURE);
	}

	if(pNode == pLinkedList->pFirstNode)
	{
		pLinkedList->pFirstNode = pNode->pNextNode;
	}
	else
	{
		struct TRM_LinkedList_Node* pPreviousNode = pLinkedList->pFirstNode;
		while(pPreviousNode->pNextNode != pNode)
		{
			pPreviousNode = pPreviousNode->pNextNode;
		}

		pPreviousNode->pNextNode = pNode->pNextNode;
	}

	free(pNode->pData);
	free(pNode);

	pLinkedList->elementCount -= 1;
}

/* Doubly Linked List */

static void TRM_DoublyLinkedList_createNode(const void* pElement, const struct TRM_DoublyLinkedList* pDoublyLinkedList, struct TRM_DoublyLinkedList_Node** ppNode)
{
	TRM_Memory_allocate(sizeof(struct TRM_DoublyLinkedList_Node), (void**)ppNode);
	(*ppNode)->pPreviousNode = NULL;
	(*ppNode)->pNextNode = NULL;
	TRM_Memory_allocate(pDoublyLinkedList->elementSize, (void**)&((*ppNode)->pData));
	TRM_Memory_memcpy(pDoublyLinkedList->elementSize, pElement, (*ppNode)->pData);
}

void TRM_DoublyLinkedList_create(size_t elementSize, struct TRM_DoublyLinkedList* pDoublyLinkedList)
{
	pDoublyLinkedList->elementSize = elementSize;
	pDoublyLinkedList->elementCount = 0;
	pDoublyLinkedList->pFirstNode = NULL;
}

void TRM_DoublyLinkedList_destroy(struct TRM_DoublyLinkedList* pDoublyLinkedList)
{
	struct TRM_DoublyLinkedList_Node* pNode = pDoublyLinkedList->pFirstNode;
	while(pNode != NULL)
	{
		struct TRM_DoublyLinkedList_Node* pNextNode = pNode->pNextNode;
		free(pNode->pData);
		free(pNode);
		pNode = pNextNode;
	}
}

void TRM_DoublyLinkedList_push(const void* pElement, struct TRM_DoublyLinkedList* pDoublyLinkedList)
{
	struct TRM_DoublyLinkedList_Node* pNewNode = NULL;
	TRM_DoublyLinkedList_createNode(pElement, pDoublyLinkedList, &pNewNode);

	if(pDoublyLinkedList->pFirstNode == NULL)
	{
		pDoublyLinkedList->pFirstNode = pNewNode;
	}
	else
	{
		struct TRM_DoublyLinkedList_Node* pNode = pDoublyLinkedList->pFirstNode;
		while(pNode->pNextNode != NULL)
		{
			pNode = pNode->pNextNode;
		}

		pNode->pNextNode = pNewNode;
		pNewNode->pPreviousNode = pNode;
	}
	
	pDoublyLinkedList->elementCount += 1;
}

void TRM_DoublyLinkedList_insertAfter(const void* pElement, struct TRM_DoublyLinkedList_Node* pNode, struct TRM_DoublyLinkedList* pDoublyLinkedList)
{
	if(pNode == NULL)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_DoublyLinkedList_Node* pNewNode = NULL;
	TRM_DoublyLinkedList_createNode(pElement, pDoublyLinkedList, &pNewNode);

	pNewNode->pPreviousNode = pNode;
	pNewNode->pNextNode = pNode->pNextNode;

	if(pNode->pNextNode != NULL)
	{
		pNode->pNextNode->pPreviousNode = pNewNode;
	}

	pNode->pNextNode = pNewNode;

	pDoublyLinkedList->elementCount += 1;
}

void TRM_DoublyLinkedList_delete(struct TRM_DoublyLinkedList_Node* pNode, struct TRM_DoublyLinkedList* pDoublyLinkedList)
{
	if(pNode == NULL)
	{
		exit(EXIT_FAILURE);
	}

	if(pNode->pPreviousNode != NULL)
	{
		pNode->pPreviousNode->pNextNode = pNode->pNextNode;
	}
	else
	{
		pDoublyLinkedList->pFirstNode = pNode->pNextNode;
	}
	
	if(pNode->pNextNode != NULL)
	{
		pNode->pNextNode->pPreviousNode = pNode->pPreviousNode;
	}

	free(pNode->pData);
	free(pNode);

	pDoublyLinkedList->elementCount -= 1;
}
