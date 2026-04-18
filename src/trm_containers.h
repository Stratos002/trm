#ifndef TRM_CONTAINERS
#define TRM_CONTAINERS

#include <stdint.h>
#include <stddef.h>

/* Dynamic Array */

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

/* Arena */

struct TRM_Arena
{
	size_t elementSize;
	uint32_t elementCapacity;
	uint32_t freeIndexCount;
	uint32_t* pFreeIndices;
	void* pData;
};

void TRM_Arena_create(size_t elementSize, uint32_t elementCapacity, struct TRM_Arena* pArena);

void TRM_Arena_destroy(struct TRM_Arena* pArena);

void TRM_Arena_add(const void* pElement, struct TRM_Arena* pArena, uint32_t* pElementIndex);

void TRM_Arena_remove(uint32_t elementIndex, struct TRM_Arena* pArena);

void TRM_Arena_get(uint32_t elementIndex, struct TRM_Arena arena, void** ppElement);

/* Linked List */

struct TRM_LinkedList_Node
{
	void* pData;
	struct TRM_LinkedList_Node* pNextNode;
};

struct TRM_LinkedList
{
	size_t elementSize;
	uint32_t elementCount;
	struct TRM_LinkedList_Node* pFirstNode;
};

void TRM_LinkedList_create(size_t elementSize, struct TRM_LinkedList* pLinkedList);

void TRM_LinkedList_destroy(struct TRM_LinkedList* pDoublyLinkedList);

void TRM_LinkedList_push(const void* pElement, struct TRM_LinkedList* pLinkedList);

void TRM_LinkedList_insertAfter(const void* pElement, struct TRM_LinkedList_Node* pNode, struct TRM_LinkedList* pLinkedList);

void TRM_LinkedList_delete(struct TRM_LinkedList_Node* pNode, struct TRM_LinkedList* pLinkedList);

/* Doubly Linked List */

struct TRM_DoublyLinkedList_Node
{
	void* pData;
	struct TRM_DoublyLinkedList_Node* pPreviousNode;
	struct TRM_DoublyLinkedList_Node* pNextNode;
};

struct TRM_DoublyLinkedList
{
	size_t elementSize;
	uint32_t elementCount;
	struct TRM_DoublyLinkedList_Node* pFirstNode;
};

void TRM_DoublyLinkedList_create(size_t elementSize, struct TRM_DoublyLinkedList* pDoublyLinkedList);

void TRM_DoublyLinkedList_destroy(struct TRM_DoublyLinkedList* pDoublyLinkedList);

void TRM_DoublyLinkedList_push(const void* pElement, struct TRM_DoublyLinkedList* pDoublyLinkedList);

void TRM_DoublyLinkedList_insertAfter(const void* pElement, struct TRM_DoublyLinkedList_Node* pNode, struct TRM_DoublyLinkedList* pDoublyLinkedList);

void TRM_DoublyLinkedList_delete(struct TRM_DoublyLinkedList_Node* pNode, struct TRM_DoublyLinkedList* pDoublyLinkedList);

#endif
