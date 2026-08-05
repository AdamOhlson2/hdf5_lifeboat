/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the LICENSE file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose:	Operations on the global heap.  The global heap is the set of
 *		all collections and each collection contains one or more
 *		global heap objects.  An object belongs to exactly one
 *		collection.  A collection is treated as an atomic entity for
 *		the purposes of I/O and caching.
 *
 *		Each file has a small cache of global heap collections called
 *		the CWFS list and recently accessed collections with free
 *		space appear on this list.  As collections are accessed the
 *		collection is moved toward the front of the list.  New
 *		collections are added to the front of the list while old
 *		collections are added to the end of the list.
 *
 *		The collection model reduces the overhead which would be
 *		incurred if the global heap were a single object, and the
 *		CWFS list allows the library to cheaply choose a collection
 *		for a new object based on object size, amount of free space
 *		in the collection, and temporal locality.
 */

/****************/
/* Module Setup */
/****************/

#include "H5HGmodule.h" /* This source code file is part of the H5HG module */

/***********/
/* Headers */
/***********/
#include "H5private.h"   /* Generic Functions			*/
#include "H5Eprivate.h"  /* Error handling		  	*/
#include "H5Fprivate.h"  /* File access				*/
#include "H5FLprivate.h" /* Free Lists                               */
#include "H5HGpkg.h"     /* Global heaps				*/
#include "H5MFprivate.h" /* File memory management		*/
#include "H5MMprivate.h" /* Memory management			*/

/****************/
/* Local Macros */
/****************/

/*
 * The maximum number of links allowed to a global heap object.
 */
#define H5HG_MAXLINK 65535

/*
 * The maximum number of indices allowed in a global heap object.
 */
#define H5HG_MAXIDX 65535

/******************/
/* Local Typedefs */
/******************/

/********************/
/* Package Typedefs */
/********************/

/********************/
/* Local Prototypes */
/********************/

static haddr_t H5HG__create(H5F_t *f, size_t size);
static size_t  H5HG__alloc(H5F_t *f, H5HG_heap_t *heap, size_t size, unsigned *heap_flags_ptr);

/* Chunk-local heap helper routine */
H5HG_heap_t *H5HG__create_local(H5F_t *f, size_t init_size);

/*********************/
/* Package Variables */
/*********************/

/* Package initialization variable */
bool H5_PKG_INIT_VAR = false;

/* Declare a free list to manage the H5HG_heap_t struct */
H5FL_DEFINE(H5HG_heap_t);

/* Declare a free list to manage sequences of H5HG_obj_t's */
H5FL_SEQ_DEFINE(H5HG_obj_t);

/* Declare a PQ free list to manage heap chunks */
H5FL_BLK_DEFINE(gheap_chunk);

/*****************************/
/* Library Private Variables */
/*****************************/

/*******************/
/* Local Variables */
/*******************/

/*-------------------------------------------------------------------------
 * Function:	H5HG__create
 *
 * Purpose:	Creates a global heap collection of the specified size.  If
 *		SIZE is less than some minimum it will be readjusted.  The
 *		new collection is allocated in the file and added to the
 *		beginning of the CWFS list.
 *
 * Return:	Success:	Ptr to a cached heap.  The pointer is valid
 *				only until some other hdf5 library function
 *				is called.
 *
 *		Failure:	NULL
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5HG__create(H5F_t *f, size_t size)
{
    H5HG_heap_t *heap = NULL;
    uint8_t     *p    = NULL;
    haddr_t      addr = HADDR_UNDEF;
    size_t       n;
    haddr_t      ret_value = HADDR_UNDEF; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    if (size < H5HG_MINSIZE)
        size = H5HG_MINSIZE;
    size = H5HG_ALIGN(size);

    /* Create it */
    H5_CHECK_OVERFLOW(size, size_t, hsize_t);
    if (HADDR_UNDEF == (addr = H5MF_alloc(f, H5FD_MEM_GHEAP, (hsize_t)size)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF, "unable to allocate file space for global heap");
    if (NULL == (heap = H5FL_CALLOC(H5HG_heap_t)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");
    heap->addr   = addr;
    heap->size   = size;
    heap->shared = H5F_SHARED(f);

    if (NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");
    memset(heap->chunk, 0, size);
    heap->nalloc = H5HG_NOBJS(f, size);
    heap->nused  = 1; /* account for index 0, which is used for the free object */
    if (NULL == (heap->obj = H5FL_SEQ_MALLOC(H5HG_obj_t, heap->nalloc)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");

    /* Initialize the header */
    H5MM_memcpy(heap->chunk, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC);
    p    = heap->chunk + H5_SIZEOF_MAGIC;
    *p++ = H5HG_VERSION;
    *p++ = 0; /*reserved*/
    *p++ = 0; /*reserved*/
    *p++ = 0; /*reserved*/
    H5F_ENCODE_LENGTH(f, p, size);

    /*
     * Padding so free space object is aligned. If malloc returned memory
     * which was always at least H5HG_ALIGNMENT aligned then we could just
     * align the pointer, but this might not be the case.
     */
    n = (size_t)H5HG_ALIGN(p - heap->chunk) - (size_t)(p - heap->chunk);
    p += n;

    /* The freespace object */
    heap->obj[0].size = size - H5HG_SIZEOF_HDR(f);
    assert(H5HG_ISALIGNED(heap->obj[0].size));
    heap->obj[0].nrefs = 0;
    heap->obj[0].begin = p;
    UINT16ENCODE(p, 0); /*object ID*/
    UINT16ENCODE(p, 0); /*reference count*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);

    /* Add this heap to the beginning of the CWFS list */
    if (H5F_cwfs_add(f, heap) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF,
                    "unable to add global heap collection to file's CWFS");

    /* Add the heap to the cache */
    if (H5AC_insert_entry(f, H5AC_GHEAP, addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF, "unable to cache global heap collection");

    ret_value = addr;

done:
    /* Cleanup on error */
    if (!H5_addr_defined(ret_value)) {
        if (H5_addr_defined(addr)) {
            /* Release the space on disk */
            if (H5MF_xfree(f, H5FD_MEM_GHEAP, addr, (hsize_t)size) < 0)
                HDONE_ERROR(H5E_BTREE, H5E_CANTFREE, HADDR_UNDEF, "unable to free global heap");

            /* Check if the heap object was allocated */
            if (heap)
                /* Destroy the heap object */
                if (H5HG__free(heap) < 0)
                    HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, HADDR_UNDEF,
                                "unable to destroy global heap collection");
        } /* end if */
    }     /* end if */

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__create() */

/*-------------------------------------------------------------------------
 * Function:	H5HG__protect
 *
 * Purpose:	Convenience wrapper around H5AC_protect on an indirect block
 *
 * Return:	Pointer to indirect block on success, NULL on failure
 *
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__protect(H5F_t *f, haddr_t addr, unsigned flags)
{
    H5HG_heap_t *heap;             /* Global heap */
    H5HG_heap_t *ret_value = NULL; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(f);
    assert(H5_addr_defined(addr));

    /* only H5AC__READ_ONLY_FLAG may appear in flags */
    assert((flags & (unsigned)(~H5AC__READ_ONLY_FLAG)) == 0);

    /* Lock the heap into memory */
    if (NULL == (heap = (H5HG_heap_t *)H5AC_protect(f, H5AC_GHEAP, addr, f, flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, NULL, "unable to protect global heap");

    /* Set the heap's address */
    heap->addr = addr;

    /* Set the return value */
    ret_value = heap;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__protect() */

/*-------------------------------------------------------------------------
 * Function:	H5HG__alloc
 *
 * Purpose:	Given a heap with enough free space, this function will split
 *		the free space to make a new empty heap object and initialize
 *		the header.  SIZE is the exact size of the object data to be
 *		stored. It will be increased to make room for the object
 *		header and then rounded up for alignment.
 *
 * Return:	Success:	The heap object ID of the new object.
 *
 *		Failure:	0
 *
 *-------------------------------------------------------------------------
 */
static size_t
H5HG__alloc(H5F_t *f, H5HG_heap_t *heap, size_t size, unsigned *heap_flags_ptr)
{
    size_t   idx;
    uint8_t *p;
    size_t   need      = H5HG_SIZEOF_OBJHDR(f) + H5HG_ALIGN(size);
    size_t   ret_value = 0; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(heap);
    assert(heap->obj[0].size >= need);
    assert(heap_flags_ptr);

    /*
     * Find an ID for the new object. ID zero is reserved for the free space
     * object.
     */
    if (heap->nused <= H5HG_MAXIDX)
        idx = heap->nused++;
    else {
        for (idx = 1; idx < heap->nused; idx++)
            if (NULL == heap->obj[idx].begin)
                break;
    } /* end else */

    assert(idx < heap->nused);

    /* Check if we need more room to store heap objects */
    if (idx >= heap->nalloc) {
        size_t      new_alloc; /* New allocation number */
        H5HG_obj_t *new_obj;   /* New array of object descriptions */

        /* Determine the new number of objects to index */
        /* nalloc is *not* guaranteed to be a power of 2! - NAF 10/26/09 */
        new_alloc = MIN(MAX(heap->nalloc * 2, (idx + 1)), (H5HG_MAXIDX + 1));
        assert(idx < new_alloc);

        /* Reallocate array of objects */
        if (NULL == (new_obj = H5FL_SEQ_REALLOC(H5HG_obj_t, heap->obj, new_alloc)))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, 0, "memory allocation failed");

        /* Clear newly allocated space */
        memset(&new_obj[heap->nalloc], 0, (new_alloc - heap->nalloc) * sizeof(heap->obj[0]));

        /* Update heap information */
        heap->nalloc = new_alloc;
        heap->obj    = new_obj;
        assert(heap->nalloc > heap->nused);
    } /* end if */

    /* Initialize the new object */
    heap->obj[idx].nrefs = 0;
    heap->obj[idx].size  = size;
    heap->obj[idx].begin = heap->obj[0].begin;
    p                    = heap->obj[idx].begin;
    UINT16ENCODE(p, idx);
    UINT16ENCODE(p, 0); /*nrefs*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, size);

    /* Fix the free space object */
    if (need == heap->obj[0].size) {
        /*
         * All free space has been exhausted from this collection.
         */
        heap->obj[0].size  = 0;
        heap->obj[0].begin = NULL;
    } /* end if */
    else if (heap->obj[0].size - need >= H5HG_SIZEOF_OBJHDR(f)) {
        /*
         * Some free space remains and it's larger than a heap object header,
         * so write the new free heap object header to the heap.
         */
        heap->obj[0].size -= need;
        heap->obj[0].begin += need;
        p = heap->obj[0].begin;
        UINT16ENCODE(p, 0); /*id*/
        UINT16ENCODE(p, 0); /*nrefs*/
        UINT32ENCODE(p, 0); /*reserved*/
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
        assert(H5HG_ISALIGNED(heap->obj[0].size));
    } /* end else-if */
    else {
        /*
         * Some free space remains but it's smaller than a heap object header,
         * so we don't write the header.
         */
        heap->obj[0].size -= need;
        heap->obj[0].begin += need;
        assert(H5HG_ISALIGNED(heap->obj[0].size));
    }

    /* Mark the heap as dirty */
    *heap_flags_ptr |= H5AC__DIRTIED_FLAG;

    /* Set the return value */
    ret_value = idx;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__alloc() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_extend
 *
 * Purpose:	Extend a heap to hold an object of SIZE bytes.
 *		SIZE is the exact size of the object data to be
 *		stored. It will be increased to make room for the object
 *		header and then rounded up for alignment.
 *
 * Return:	Success:	Non-negative
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_extend(H5F_t *f, haddr_t addr, size_t need)
{
    H5HG_heap_t *heap       = NULL;               /* Pointer to heap to extend */
    unsigned     heap_flags = H5AC__NO_FLAGS_SET; /* Flags to unprotecting heap */
    size_t       old_size;                        /* Previous size of the heap's chunk */
    uint8_t     *new_chunk;                       /* Pointer to new chunk information */
    uint8_t     *p;                               /* Pointer to raw heap info */
    unsigned     u;                               /* Local index variable */
    herr_t       ret_value = SUCCEED;             /* Return value */

    FUNC_ENTER_NOAPI_NOINIT

    /* Check args */
    assert(f);
    assert(H5_addr_defined(addr));

    /* Protect the heap */
    if (NULL == (heap = H5HG__protect(f, addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Re-allocate the heap information in memory */
    if (NULL == (new_chunk = H5FL_BLK_REALLOC(gheap_chunk, heap->chunk, (heap->size + need))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "new heap allocation failed");
    memset(new_chunk + heap->size, 0, need);

    /* Adjust the size of the heap */
    old_size = heap->size;
    heap->size += need;

    /* Encode the new size of the heap */
    p = new_chunk + H5_SIZEOF_MAGIC + 1 /* version */ + 3 /* reserved */;
    H5F_ENCODE_LENGTH(f, p, heap->size);

    /* Move the pointers to the existing objects to their new locations */
    for (u = 0; u < heap->nused; u++)
        if (heap->obj[u].begin)
            heap->obj[u].begin = new_chunk + (heap->obj[u].begin - heap->chunk);

    /* Update the heap chunk pointer now */
    heap->chunk = new_chunk;

    /* Update the free space information for the heap  */
    heap->obj[0].size += need;
    if (heap->obj[0].begin == NULL)
        heap->obj[0].begin = heap->chunk + old_size;
    p = heap->obj[0].begin;
    UINT16ENCODE(p, 0); /*id*/
    UINT16ENCODE(p, 0); /*nrefs*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    assert(H5HG_ISALIGNED(heap->obj[0].size));

    /* Resize the heap in the cache */
    if (H5AC_resize_entry(heap, heap->size) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTRESIZE, FAIL, "unable to resize global heap in cache");

    /* Mark the heap as dirty */
    heap_flags |= H5AC__DIRTIED_FLAG;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, heap->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to unprotect heap");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG_extend() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_insert
 *
 * Purpose:	A new object is inserted into the global heap.  It will be
 *		placed in the first collection on the CWFS list which has
 *		enough free space and that collection will be advanced one
 *		position in the list.  If no collection on the CWFS list has
 *		enough space then  a new collection will be created.
 *
 *		It is legal to push a zero-byte object onto the heap to get
 *		the reference count features of heap objects.
 *
 * Return:	Success:	Non-negative, and a heap object handle returned
 *				through the HOBJ pointer.
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_insert(H5F_t *f, size_t size, const void *obj, H5HG_t *hobj /*out*/)
{
    size_t       need; /*total space needed for object		*/
    size_t       idx;
    haddr_t      addr; /* Address of heap to add object within */
    H5HG_heap_t *heap       = NULL;
    unsigned     heap_flags = H5AC__NO_FLAGS_SET;
    herr_t       ret_value  = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(0 == size || obj);
    assert(hobj);

    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Find a large enough collection on the CWFS list */
    need = H5HG_SIZEOF_OBJHDR(f) + H5HG_ALIGN(size);

    /* Look for a heap in the file's CWFS that has enough space for the object */
    addr = HADDR_UNDEF;
    if (H5F_cwfs_find_free_heap(f, need, &addr) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_NOTFOUND, FAIL, "error trying to locate heap");

    /*
     * If we didn't find any collection with enough free space then allocate a
     * new collection large enough for the message plus the collection header.
     */
    if (!H5_addr_defined(addr)) {
        addr = H5HG__create(f, need + H5HG_SIZEOF_HDR(f));

        if (!H5_addr_defined(addr))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, FAIL, "unable to allocate a global heap collection");
    } /* end if */
    assert(H5_addr_defined(addr));

    if (NULL == (heap = H5HG__protect(f, addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Split the free space to make room for the new object */
    if (0 == (idx = H5HG__alloc(f, heap, size, &heap_flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL, "unable to allocate global heap object");

    /* Copy data into the heap */
    if (size > 0)
        H5MM_memcpy(heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f), obj, size);
    heap_flags |= H5AC__DIRTIED_FLAG;

    /* Return value */
    hobj->addr = heap->addr;
    hobj->idx  = idx;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, heap->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to unprotect heap.");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* H5HG_insert() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_read
 *
 * Purpose:	Reads the specified global heap object into the buffer OBJECT
 *		supplied by the caller.  If the caller doesn't supply a
 *		buffer then one will be allocated.  The buffer should be
 *		large enough to hold the result.
 *
 * Return:	Success:	The buffer containing the result.
 *
 *		Failure:	NULL
 *
 *-------------------------------------------------------------------------
 */
void *
H5HG_read(H5F_t *f, H5HG_t *hobj, void *object /*out*/, size_t *buf_size)
{
    H5HG_heap_t *heap = NULL;          /* Pointer to global heap object */
    size_t       size;                 /* Size of the heap object */
    uint8_t     *p;                    /* Pointer to object in heap buffer */
    void        *orig_object = object; /* Keep a copy of the original object pointer */
    void        *ret_value   = NULL;   /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, NULL)

    /* Check args */
    assert(f);
    assert(hobj);

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__READ_ONLY_FLAG)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, NULL, "unable to protect global heap");
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    size = heap->obj[hobj->idx].size;
    p    = heap->obj[hobj->idx].begin + H5HG_SIZEOF_OBJHDR(f);

    /* Allocate a buffer for the object read in, if the user didn't give one */
    if (!object && NULL == (object = H5MM_malloc(size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");
    H5MM_memcpy(object, p, size);

    /*
     * Advance the heap in the CWFS list. We might have done this already
     * with the H5AC_protect(), but it won't hurt to do it twice.
     */
    if (heap->obj[0].begin) {
        if (H5F_cwfs_advance_heap(f, heap, false) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTMODIFY, NULL, "can't adjust file's CWFS");
    } /* end if */

    /* If the caller would like to know the heap object's size, set that */
    if (buf_size)
        *buf_size = size;

    /* Set return value */
    ret_value = object;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, NULL, "unable to release object header");

    if (NULL == ret_value && NULL == orig_object && object)
        H5MM_free(object);

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_read() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_link
 *
 * Purpose:	Adjusts the link count for a global heap object by adding
 *		ADJUST to the current value.  This function will fail if the
 *		new link count would overflow.  Nothing special happens when
 *		the link count reaches zero; in order for a heap object to be
 *		removed one must call H5HG_remove().
 *
 * Return:	Success:	Number of links present after the adjustment.
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
int
H5HG_link(H5F_t *f, const H5HG_t *hobj, int adjust)
{
    H5HG_heap_t *heap       = NULL;
    unsigned     heap_flags = H5AC__NO_FLAGS_SET;
    int          ret_value  = -1; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    if (adjust != 0) {
        if (hobj->idx >= heap->nused)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                        hobj->addr, hobj->idx);
        if (NULL == heap->obj[hobj->idx].begin)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                        hobj->addr, hobj->idx);
        if ((heap->obj[hobj->idx].nrefs + adjust) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "new link count would be out of range");
        if ((heap->obj[hobj->idx].nrefs + adjust) > H5HG_MAXLINK)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "new link count would be out of range");
        heap->obj[hobj->idx].nrefs += adjust;
        heap_flags |= H5AC__DIRTIED_FLAG;
    } /* end if */

    /* Set return value */
    ret_value = heap->obj[hobj->idx].nrefs;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_link() */

/*-------------------------------------------------------------------------
 * Function:    H5HG_get_obj_size
 *
 * Purpose:     Returns the size of a global heap object.
 * Return:      Success:        Non-negative
 *
 *              Failure:        Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_get_obj_size(H5F_t *f, H5HG_t *hobj, size_t *obj_size)
{
    H5HG_heap_t *heap      = NULL;    /* Pointer to global heap object */
    herr_t       ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    assert(obj_size);

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__READ_ONLY_FLAG)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Sanity check the heap object */
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Set object size */
    *obj_size = heap->obj[hobj->idx].size;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_get_obj_size() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_remove
 *
 * Purpose:	Removes the specified object from the global heap.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_remove(H5F_t *f, H5HG_t *hobj)
{
    H5HG_heap_t *heap = NULL;
    uint8_t     *p = NULL, *obj_start = NULL;
    size_t       need;
    unsigned     u;
    unsigned     flags     = H5AC__NO_FLAGS_SET; /* Whether the heap gets deleted */
    herr_t       ret_value = SUCCEED;            /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Sanity check the heap object (split around bugfix below) */
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* When the application selects the same location to rewrite the VL element by using H5Sselect_elements,
     * it can happen that the entry has been removed by first rewrite.  Here we simply skip the removal of
     * the entry and let the second rewrite happen (see HDFFV-10635).  In the future, it'd be nice to handle
     * this situation in H5T_conv_vlen in H5Tconv.c instead of this level (HDFFV-10648). */
    if (heap->obj[hobj->idx].nrefs == 0 && heap->obj[hobj->idx].size == 0 && !heap->obj[hobj->idx].begin)
        HGOTO_DONE(SUCCEED);

    /* Finish sanity checking the heap object */
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    obj_start = heap->obj[hobj->idx].begin;
    /* Include object header size */
    need = H5HG_ALIGN(heap->obj[hobj->idx].size) + H5HG_SIZEOF_OBJHDR(f);

    /* Move the new free space to the end of the heap */
    for (u = 0; u < heap->nused; u++)
        if (heap->obj[u].begin > heap->obj[hobj->idx].begin)
            heap->obj[u].begin -= need;
    if (NULL == heap->obj[0].begin) {
        heap->obj[0].begin = heap->chunk + (heap->size - need);
        heap->obj[0].size  = need;
        heap->obj[0].nrefs = 0;
    } /* end if */
    else
        heap->obj[0].size += need;
    memmove(obj_start, obj_start + need, heap->size - (size_t)((obj_start + need) - heap->chunk));
    if (heap->obj[0].size >= H5HG_SIZEOF_OBJHDR(f)) {
        p = heap->obj[0].begin;
        UINT16ENCODE(p, 0); /*id*/
        UINT16ENCODE(p, 0); /*nrefs*/
        UINT32ENCODE(p, 0); /*reserved*/
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    } /* end if */
    memset(heap->obj + hobj->idx, 0, sizeof(H5HG_obj_t));
    flags |= H5AC__DIRTIED_FLAG;

    if ((heap->obj[0].size + H5HG_SIZEOF_HDR(f)) == heap->size) {
        /*
         * The collection is empty. Remove it from the CWFS list and return it
         * to the file free list.
         */
        flags |=
            H5AC__DELETED_FLAG |
            H5AC__FREE_FILE_SPACE_FLAG; /* Indicate that the object was deleted, for the unprotect call */
    }                                   /* end if */
    else {
        /*
         * If the heap is in the CWFS list then advance it one position.  The
         * H5AC_protect() might have done that too, but that's okay.  If the
         * heap isn't on the CWFS list then add it to the end.
         */
        if (H5F_cwfs_advance_heap(f, heap, true) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTMODIFY, FAIL, "can't adjust file's CWFS");
    } /* end else */

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_remove() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__free
 *
 * Purpose:     Destroys a global heap collection in memory
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__free(H5HG_heap_t *heap)
{
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(heap);

    /* Remove the heap from the CWFS list */
    if (H5F_cwfs_remove_heap(heap->shared, heap) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTREMOVE, FAIL, "can't remove heap from file's CWFS");

    if (heap->chunk)
        heap->chunk = H5FL_BLK_FREE(gheap_chunk, heap->chunk);
    if (heap->obj)
        heap->obj = H5FL_SEQ_FREE(H5HG_obj_t, heap->obj);
    heap = H5FL_FREE(H5HG_heap_t, heap);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__free() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__free_local
 *
 * Purpose:     Frees a chunk-local heap object.
 *
 *              This differs from H5HG__free() because the chunk-local heap is
 *              not on the CWFS list and is not a standalone metadata-cache
 *              object. It is owned by the decoded structured chunk.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                          --AZO    07/12/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__free_local(H5HG_heap_t *heap)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(heap);

    /* Release the serialized H5HG collection image owned by this local heap. */
    if (heap->chunk)
        heap->chunk = H5FL_BLK_FREE(gheap_chunk, heap->chunk);

    /*
     * Release the in-memory object table used to map local indices to records
     * within the serialized heap image
     */
    if (heap->obj)
        heap->obj = H5FL_SEQ_FREE(H5HG_obj_t, heap->obj);

    /*
     * Release the heap descriptor itself. No CWFS removal is required because
     * local heaps are never registered with the file-wide global heap machinery.
     */
    heap = H5FL_FREE(H5HG_heap_t, heap);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__free_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__create_local
 *
 * Purpose:     Creates an in-memory H5HG-style heap for use as a
 *              chunk-local VL heap.
 *
 *              Unlike H5HG__create(), this routine does not allocate
 *              a standalone global-heap collection in the file, assign
 *              a file address, add the heap to the CWFS list, or insert
 *              it into the metadata cache. The returned heap is owned
 *              by the decoded structured chunk and will later be
 *              serialized into H5_SECTION_VL.
 *
 *              The heap image uses the existing H5HG collection layout.
 *              Object-table index zero is initialized as the free-space
 *              record, and indices beginning at one are available for
 *              payload objects.
 *
 * Return:      Success: Pointer to the created local heap
 *              Failure: NULL
 *
 *                                                  -- AZO  6/26/26
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__create_local(H5F_t *f, size_t init_size)
{
    H5HG_heap_t *heap         = NULL;      /* New local heap being constructed */
    uint8_t     *p            = NULL;      /* Current position in encoded heap image */
    size_t       size         = init_size; /* Requested/adjusted heap image size */
    size_t       aligned_size = 0;         /* Result of safely aligning SIZE */
    size_t       pad_size     = 0;         /* Padding before the first heap object */
    H5HG_heap_t *ret_value    = NULL;      /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);

    /*
     * Use the normal min H5HG collection size so the local heap has
     * enough room for its header, free-space record, and initial objects.
     */
    if ((size < H5HG_MINSIZE))
        size = H5HG_MINSIZE;

    /*
     * Align the complete heap image into the H5HG allocation boundary.
     * A wrapped alignment result would be smaller than the requested size.
     */
    aligned_size = H5HG_ALIGN(size);
    if ((aligned_size < size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap initial size overflow");

    size = aligned_size;

    /* The heap must be large enough to contain its collection header. */
    if ((size < H5HG_SIZEOF_HDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap size is smaller than its header");

    /* Allocate and initialize the in-memory heap descriptor. */
    if ((NULL == (heap = H5FL_CALLOC(H5HG_heap_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    heap->addr = HADDR_UNDEF;
    heap->size = size;

    /*
     * The local heap may use file-width information while the decoded
     * structured chunk is alive, but it is never placed on the file CWFS
     * list or in the metadata cache.
     */
    heap->shared = H5F_SHARED(f);

    /* Allocate and clear the serialized H5HG collection image. */
    if ((NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    memset(heap->chunk, 0, size);

    /*
     * Allocate the in-memory object table. Entry zero is reserved for the
     * free-space record; payload objects begin at index one.
     */
    heap->nalloc = H5HG_NOBJS(f, size);
    heap->nused  = 1; /* Index zero is the free-space object */
    heap->nlive  = 0; /* No live payload objects have been inserted */

    if ((0 == heap->nalloc))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "invalid local heap object-table size");

    if ((heap->nalloc > (SIZE_MAX / sizeof(heap->obj[0]))))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "local heap object-table size overflow");

    if ((NULL == (heap->obj = H5FL_SEQ_MALLOC(H5HG_obj_t, heap->nalloc))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    memset(heap->obj, 0, heap->nalloc * sizeof(heap->obj[0]));

    /*
     * Initialize the collection header:
     * (signature, version, three reserved bytes, encoded collection size)
     */
    H5MM_memcpy(heap->chunk, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC);

    p    = heap->chunk + H5_SIZEOF_MAGIC;
    *p++ = H5HG_VERSION;
    *p++ = 0; /* Reserved */
    *p++ = 0; /* Reserved */
    *p++ = 0; /* Reserved */

    H5F_ENCODE_LENGTH(f, p, size);

    /*
     * Align the beginning of object zero in the same way as a normal H5HG
     * collection.
     */
    pad_size = (size_t)H5HG_ALIGN(p - heap->chunk) - (size_t)(p - heap->chunk);
    p += pad_size;

    /* Initialize object zero as the free-space object */
    heap->obj[0].size  = size - H5HG_SIZEOF_HDR(f);
    heap->obj[0].nrefs = 0;
    heap->obj[0].begin = p;

    assert(H5HG_ISALIGNED(heap->obj[0].size));

    /* Write the serialized free-space object header into the heap image. */
    UINT16ENCODE(p, 0); /* Object ID */
    UINT16ENCODE(p, 0); /* Reference count */
    UINT32ENCODE(p, 0); /* Reserved */
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);

    ret_value = heap;

done:
    /* Destroy any partially initialized heap if creation fails.  */
    if ((NULL == ret_value && heap))
        if ((H5HG__free_local(heap) < 0))
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL, "unable to destroy local heap");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__create_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__extend_local
 *
 * Purpose:     Extends a chunk-local H5HG-style heap's in-memory byte
 *              buffer by at least NEED bytes.
 *
 *              Unlike H5HG_extend(), this routine operates directly on a
 *              heap owned by the current structured chunk. It does not
 *              locate or protect a standalone global-heap collection,
 *              resize a metadata-cache entry, or perform any file-space
 *              or CWFS operation.
 *
 *              Existing object-table pointers are preserved across the
 *              buffer reallocation, the encoded heap size is updated,
 *              and the newly appended bytes are added to object zero,
 *              which represents the heap's free-space extent.
 *
 *              NEED is the minimum additional free space required, not
 *              the final heap size. A value of zero is accepted as a
 *              successful no-op.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                                  -- AZO   7/08/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__extend_local(H5F_t *f, H5HG_heap_t *heap, size_t need)
{
    size_t  *obj_offsets = NULL;  /* Object offsets */
    size_t   old_size    = 0;     /* Old size of the heap's chunk */
    size_t   ext_size    = 0;     /* Extension size */
    size_t   new_size    = 0;     /* New size of the heap's chunk */
    uint8_t *new_chunk   = NULL;  /* Pointer to new chunk information */
    uint8_t *p           = NULL;  /* Pointer to raw heap info */
    unsigned u;                   /* Local index variable */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Checl args */
    assert(f);
    assert(heap);
    assert(heap->chunk);
    assert(heap->obj);

    /* 
     * A rquest to extend the heap by zero bytes indicates an invalid or unnecessary call. Callers
     * should invoke this routine only after determining that additional heap capacity is required.
     */
    if ( 0 == need )
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "zero-byte chunk-local heap extension requested");

    /*
     * Detect overflow during alignment. H5HG_ALIGN() should never produce
     * a result smaller than its input unless the calculation wrapped.
     */
    ext_size = H5HG_ALIGN(need);
    if ( ext_size < need )
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap extension size overflow");

    /*
     * Grow by at least H5HG_MINSIZE to avoid excessive reallocations for
     * repeated small inserts.
     */
    if ((ext_size < H5HG_MINSIZE))
        ext_size = H5HG_MINSIZE;

    old_size = heap->size;

    /* Validate all size calculation before modifying the heap. */
    if ( (ext_size) > (SIZE_MAX - old_size) )
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap size overflow");

    if ( (ext_size) > (SIZE_MAX - heap->obj[0].size) )
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap free-space size overflow");

    new_size = old_size + ext_size;

    /*
     * Save object offsets before reallocating because the old allocation
     * may be moved or released by the realloc operation.
     */
    if ( heap->nused > (SIZE_MAX / sizeof(*obj_offsets)) )
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap offset-table size overflow");

    if ((heap->nused > 0)) {
        if ((NULL == (obj_offsets = H5MM_malloc(heap->nused * sizeof(*obj_offsets)))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate local heap offset table");

        for (u = 0; u < heap->nused; u++) {
            if (heap->obj[u].begin)
                obj_offsets[u] = (size_t)(heap->obj[u].begin - heap->chunk);
            else
                obj_offsets[u] = SIZE_MAX;
        }
    } /* end if */

    /*
     * Grow only the in-memory heap image. The local heap has no independent
     * file allocation or metadata-cache entry.
     */
    if ((NULL == (new_chunk = H5FL_BLK_REALLOC(gheap_chunk, heap->chunk, new_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to extend chunk-local heap buffer");

    /* Initialize the newly appended region deterministically. */
    memset(new_chunk + old_size, 0, ext_size);

    heap->chunk = new_chunk;
    heap->size  = new_size;

    /* Rebuild object pointers using the saved offsets. Cleared entries remain NULL. */
    for (u = 0; u < heap->nused; u++) {
        if (SIZE_MAX != obj_offsets[u])
            heap->obj[u].begin = heap->chunk + obj_offsets[u];
        else
            heap->obj[u].begin = NULL;
    }

    /* Update the encoded collection size */
    p = heap->chunk + H5_SIZEOF_MAGIC + 1 /* version */ + 3 /* reserved */;
    H5F_ENCODE_LENGTH(f, p, heap->size);

    /*
     * If the previous collection had no free space, the newly appended
     * region begins at the previous end of the collection.
     */
    if (NULL == heap->obj[0].begin)
        heap->obj[0].begin = heap->chunk + old_size;


    heap->obj[0].size += ext_size;
    heap->obj[0].nrefs = 0;

    /* Rewrite the serialized free-space header to reflect the enlarged extent. */
    p = heap->obj[0].begin;

    UINT16ENCODE(p, 0); /* Object ID */
    UINT16ENCODE(p, 0); /* Reference count */
    UINT32ENCODE(p, 0); /* Reserved */
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);

    assert(H5HG_ISALIGNED(heap->obj[0].size));

done:
    if (obj_offsets)
        H5MM_free(obj_offsets);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__extend_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__insert_local
 *
 * Purpose:     Inserts an object into a chunk-local heap and returns the
 *              local object index.
 *
 *              Unlike H5HG_insert(), this routine operates on the heap
 *              owned by the current structured chunk. It does not search
 *              the file-wide global heap free-space list, allococate a
 *              standalone heap collection in the file, or interact with
 *              the metadata cache.
 *
 *              If *HEAP_PTR is NULL, a new local heap is created and
 *              returned through HEAP_PTR. If the existing heap does
 *              not contain enough free space, its in-memory image is
 *              extended.
 *
 *              SIZE may be zero. OBJ must be non-NULL when SIZE is greater
 *              than zero. On success, IDX_OUT receives the object-table
 *              index used to identify the payload within this local heap.
 *
 *              The local heap remains owned by the caller. The object
 *              index is valid only in combination with that heap and
 *              does not identify a standalone global-heap collection
 *              in the file.
 *
 * Return:      SUCCEED/FAIL
 *
 *
 *                                                  -- AZO   07/12/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__insert_local(H5F_t *f, H5HG_heap_t **heap_ptr, size_t size, const void *obj, size_t *idx_out)
{
    H5HG_heap_t *heap         = NULL; /* Pointer to the heap object */
    size_t       aligned_size = 0;
    size_t       initial_size;
    size_t       need         = 0; /* Totl space needed for object */
    size_t       idx          = 0;
    unsigned     heap_flags   = H5AC__NO_FLAGS_SET;
    bool         heap_created = false;
    herr_t       ret_value    = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap_ptr);
    assert(0 == size || obj);
    assert(idx_out);

    /*
     * Ensure that callers never receive a stale index when insertion fails.
     */
    *idx_out = 0;

    aligned_size = H5HG_ALIGN(size);
    if (aligned_size < size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object size overflow");

    if (aligned_size > (SIZE_MAX - H5HG_SIZEOF_OBJHDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object allocation size overflow");

    /* Compute the complete serialized extent of the object with its header and aligned payload */
    need = H5HG_SIZEOF_OBJHDR(f) + aligned_size;

    /*
     * Lazily create the heap for the first inserted object.
     */
    if (NULL == *heap_ptr) {

        if ((need > (SIZE_MAX - H5HG_SIZEOF_HDR(f))))
            HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap initial allocation size overflow");

        initial_size = H5HG_SIZEOF_HDR(f) + need;

        /*
         * Create the chunk's local heap on demand when the first payload is stored.
         * The new heap is installed directly into the decoded chunk state.
         */
        if (NULL == (*heap_ptr = H5HG__create_local(f, initial_size)))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, FAIL, "unable to create chunk-local heap");

        heap_created = true;
    }

    heap = *heap_ptr;

    /*
     * Object zero is reserved for free space, leaving H5HG_MAXIDX possible
     * payload indices. Since nlive counts only payload objects, reaching this
     * value means that every available object index is currently in use.
     */
    if (heap->nlive >= H5HG_MAXIDX)
        HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL,
                    "chunk-local heap object index space is exhausted");

    /*
     * Extend the current heap image when object zero does not contain enough
     * free space for the complete serialized object.
     */
    if (heap->obj[0].size < need) {
        if (H5HG__extend_local(f, heap, need - heap->obj[0].size) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL,
                        "unable to extend chunk-local heap");
    }

    /*
     * Use existing H5HG allocator to select an object index, write the
     * serialized object header, and consume space from object zero.
     *
     * H5HG__alloc() may set metadata-cache flags for normal global heaps.
     * This chunk-local heap is not a metadata-cache entry, so heap_flags is
     * intentionally ignored after the call.
     */
    if (0 == (idx = H5HG__alloc(f, heap, size, &heap_flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL,
                    "unable to allocate chunk-local heap object");

    /* Copy the logical payload; alignment padding remains zero-filled. */
    if (size > 0)
        H5MM_memcpy(heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f), obj, size);

    /*
     * The object is now fully initialized and represents one additional
     * live payload. Zero-length objects are included in this count.
     */
    heap->nlive++;

    *idx_out = idx;

done:
    /*
     * If this call created the heap but failed to complete its first
     * insertion, destroy the partially initialized heap and restore the
     * caller's pointer to NULL.
     */
    if ((ret_value < 0) && heap_created && *heap_ptr) {
        if (H5HG__free_local(*heap_ptr) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL,
                        "unable to free newly created chunk-local heap");

        *heap_ptr = NULL;
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__insert_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__read_local
 *
 * Purpose:     Reads an object from a chunk-local H5HG-style heap.
 *
 *              Unlike H5HG_read(), this routine operates on an already
 *              decoded heap supplied by the caller. The heap is owned
 *              by the current structured chunk and is not located or
 *              protected through the metadata cache.
 *
 *              IDX identifies the object within the supplied heap. Object
 *              index zero is reserved for the heap's free-space record
 *              and is not valid payload object.
 *
 *              OBJECT must point to a caller-owned buffer large enough
 *              to hold the object's logical payload. This routine does 
 *              not allocate a destination buffer.
 *
 *              On success, BUF_SIZE, when non-NULL, is set to the logical
 *              payload size.
 *
 * Return:      Success: Pointer to the object buffer
 *              Failure: NULL
 *
 *                                                      -- AZO 06/28/26
 *
 *-------------------------------------------------------------------------
 */
void *
H5HG__read_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, void *object, size_t *buf_size)
{
    size_t   size = 0; /* Size of the heap object */
    uint8_t *p         = NULL;   /* Pointer to the object in the heap */
    void    *ret_value = NULL;   /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(heap->chunk);
    assert(heap->obj);

    if (buf_size)
        *buf_size = 0;

    /* The caller owns the destination buffer. Fail if no buffer is provided. */
    if (NULL == object)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, 
                    "chunk-local heap read requires a destination buffer");

    /* Heap object idx 0 is free space in the heap */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad local heap index");

    /* Verify that the supplied index identifies an entry in the local heap's object table */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "local heap index out of range");

    /*
     * Removed objects have cleared table entries. A NULL begin pointer therefore means
     * that the requested local object no longer exists.
     */
    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad local heap object pointer");

    /*
     * Object table stores the unaligned payload size. The serialized payload begins
     * immediately after the object's H5HG header.
     */
    size = heap->obj[idx].size;
    p    = heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f);

    /*
     * Copy only the logical payload bytes. Any alignment padding stored in
     * the heap image is not part of the returned object.
     */
    if (size > 0)
        H5MM_memcpy(object, p, size);

    /*
     * Return the logical payload size.
     */
    if (buf_size)
        *buf_size = size;

    ret_value = object;

done:

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__read_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__is_empty_local
 *
 * Purpose:     Checks whether a chunk-local heap contains any allocated
 *              payload objects.
 * 
 *              The heap maintains a count of live payload objects as
 *              objects are inserted, removed, and reconstructed during
 *              decode. Object zero, which represents the free-space
 *              extent is not included in this count.
 *
 * Return:      TRUE if no live payload objects remain
 *              FALSE if at least one live payload object remains.
 *
 *                                                  -- AZO   7/09/26
 *
 *-------------------------------------------------------------------------
 */
htri_t
H5HG__is_empty_local(const H5HG_heap_t *heap)
{

    htri_t ret_value;

    FUNC_ENTER_PACKAGE_NOERR

    /* Check argument */
    assert(heap);

    /* Object zero is not included in the live payload count */
    ret_value = (0 == heap->nlive);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__is_empty_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__get_obj_size_local
 *
 * Purpose:     Returns the payload size of an object stored in a
 *              chunk-local H5HG-style heap.
 *
 * Return:      Success:            Non-negative
 *
 *              Failure:            Negative
 *
 *                                                  -- AZO 7/11/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__get_obj_size_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, size_t *obj_size)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(obj_size);

    /* Ensure validation failures do not leave an old caller value behind */
    *obj_size = 0;

    /* Heap object idx 0 is the free space in the heap and should not be given out */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad chunk-local heap index");

    /* Sanity check the heap object */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap index is out of range");

    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap object has been removed or is invalid");

    /* Set object size */
    *obj_size = heap->obj[idx].size;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__get_obj_size_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__remove_local
 *
 * Purpose:     Removes an object from a chunk-local heap. The heap itself is
 *              not freed here; the caller owns heap lifetime.
 *
 *              Unlike H5HG_remove(), this routine operates on an in-memory
 *              heap owned by the current structured chunk rather than a
 *              standalone global heap collection stored in the file.
 *              The caller supplies the heap directly so no metadata-cache
 *              protection or file-space management is performed.
 *
 *              If heap_empty is non-null. it is set to indicate whether
 *              any allocated objects remain agter the removal completes.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 *                                                   -- AZO   7/11/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__remove_local(H5F_t *f, H5HG_heap_t *heap, size_t idx, hbool_t *heap_empty)
{
    uint8_t *p         = NULL;    /* Pointer into encoded heap image */
    uint8_t *obj_start = NULL;    /* Beginning of object being removed */
    size_t   aligned_size;        /* Aligned payload size */
    size_t   need      = 0;       /* Complete serialized object extent */
    size_t   move_size = 0;       /* Bytes shifted during compaction */
    unsigned u;                   /* Object-table index */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);

    /*
     * Initialize the optional heap-empty result so callers do not observe a
     * stale value if validation fails before the heap is modified.
     */
    if (heap_empty)
        *heap_empty = false;

    /* Heap object idx 0 is the free space in the heap and should not be shared */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad local heap index");

    /* Sanity check the heap object */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap index out of range");

    /*
     * Treat removal of an already-cleared object as a successful no-op.
     * This can occur when an overwrite path attempts to release the same 
     * old descriptor more than once.
     */
    if ((0 == heap->obj[idx].nrefs) && (0 == heap->obj[idx].size) && (NULL == heap->obj[idx].begin)) {

        if (heap_empty) {
            *heap_empty = (0 == heap->nlive);
        }

        HGOTO_DONE(SUCCEED);
    } /* end if */

    /* Finish sanity checking the heap object */
    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad local heap object pointer");

    /* A live object-table entry must correspond to at least one object in the maintained
     * live-payload count. 
     */
    if (0 == heap->nlive)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, 
                    "chunk-local heap live-count is inconsistent");

    obj_start = heap->obj[idx].begin;

    /* Compute the aligned payload size. */
    aligned_size = H5HG_ALIGN(heap->obj[idx].size);

    /* Ensure size is in valid range */
    if (aligned_size < heap->obj[idx].size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object size overflow");

    if (aligned_size > SIZE_MAX - H5HG_SIZEOF_OBJHDR(f))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap removal size overflow");

    /*
     * Compute the serialized extent occupied by this object. The extent
     * includes both the object header and the aligned payload because
     * the entire record will be removed from the heap image.
     */
    need = aligned_size + H5HG_SIZEOF_OBJHDR(f);

    /*
     * Verify that the object extent lies entirely inside the local
     * heap image. This becomes particularly important after local
     * heaps are decoded from on-disk section data.
     */
    if ((obj_start < heap->chunk) || (obj_start >= (heap->chunk + heap->size)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap object pointer is outside heap image");

    if (need > (heap->size - (size_t)(obj_start - heap->chunk)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap object extends beyond heap image");

    /*
     * Validate the new free-space size before modifying object pointers,
     * object zero, or the serialized heap image.
     */
    if (need > (SIZE_MAX - heap->obj[0].size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL,
                    "chunk-local heap free-space size overflow");

    /* The previous bounds checks guarantee that this subtraction cannot underflow */
    move_size = (heap->size - (size_t)((obj_start + need) - heap->chunk));

    /*
     * Compact the heap by shifting every object that follows the removed
     * record toward the beginning of the heap image. Update each object's
     * cached pointer before moving the serialized bytes.
     */
    for (u = 0; u < heap->nused; u++)
        if ((heap->obj[u].begin) && (heap->obj[u].begin > obj_start))
            heap->obj[u].begin -= need;

    /*
     * Add the reclaimed extent to object zero. If no serialized free-space
     * record currently exists, create one at the new end of the used region.
     */
    if (NULL == heap->obj[0].begin) {
        heap->obj[0].begin = heap->chunk + (heap->size - need);
        heap->obj[0].size  = need;
        heap->obj[0].nrefs = 0;
    }
    else {

        heap->obj[0].size += need;
    }

    /*
     * Remove the serialized object record from the heap image by sliding
     * the remaining bytes towards the beginning of the heap.
     */
    memmove(obj_start, obj_start + need, move_size);

    /*
     * Rewrite the free-space object's serialized header so the heap
     * image remains internally consistent after compaction.
     */
    if ((heap->obj[0].size >= H5HG_SIZEOF_OBJHDR(f))) {
        p = heap->obj[0].begin;

        UINT16ENCODE(p, 0); /* Object ID */
        UINT16ENCODE(p, 0); /* Reference count */
        UINT32ENCODE(p, 0); /* Reserved */
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    }

    /* Clear the in-memory object-table entry so its index may be reused. */
    memset(heap->obj + idx, 0, sizeof(H5HG_obj_t));

    /* The removal is complete. Update the maintained live-payload count. */
    heap->nlive--;

    if (heap_empty) {

        *heap_empty = (0 == heap->nlive);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__remove_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__encode_local
 *
 * Purpose:     Creates a serialized copy of a chunk-local H5HG-style heap.
 *
 *              The H5HG heap-management routines maintain HEAP->CHUNK
 *              directly in the existing serialized global-heap collection
 *              format. Consequently, this routine does not rebuild the heap
 *              record by record. It allocates an output buffer, copies the
 *              current encoded image, and returns the image and its length
 *              to the caller.
 *
 *              On success, *IMAGE_OUT points to a buffer allocated with
 *              H5MM_malloc(), and *IMAGE_LEN_OUT contains its size. The
 *              caller is responsible for releasing the buffer with
 *              H5MM_free().
 *
 *              This routine exports only the embedded H5HG collection
 *              image. The structured-chunk layer remains responsible for
 *              any H5_SECTION_VL implementation header, version, flags,
 *              checksum, or other section-level information.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   7/14/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__encode_local(const H5HG_heap_t *heap, uint8_t **image_out, size_t *image_len_out)
{
    uint8_t *image     = NULL;    /* Newly allocated serialized heap image */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(heap);
    assert(image_out);
    assert(image_len_out);

    /*
     * Initialize both outputs so a failed encode cannot leave stale values
     * in the caller.
     */
    *image_out     = NULL;
    *image_len_out = 0;

    /*
     * A chunk-local heap is embedded in its owning structured chunk and
     * therefore must not have a standalone global-heap file address.
     */
    if (H5_addr_defined(heap->addr))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap unexpectedly has a file address");

    if (NULL == heap->chunk)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap has no serialized image");

    if (heap->size < (size_t)H5HG_MINSIZE)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL,
                    "chunk-local heap image is smaller than the minimum heap size");

    /*
     * Allocate an independent output image. SCC may retain or transform this
     * buffer after the decoded in-memory heap has been released.
     */
    if (NULL == (image = H5MM_malloc(heap->size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate encoded chunk-local heap image");

    /*
     * HEAP->CHUNK is already maintained in serialized H5HG form.
     */
    H5MM_memcpy(image, heap->chunk, heap->size);

    *image_out     = image;
    *image_len_out = heap->size;
    image          = NULL;

done:
    if (image)
        H5MM_free(image);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__encode_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__decode_local
 *
 * Purpose:     Reconstructs an in-memory chunk-local H5HG-style heap from
 *              a serialized H5HG collection image.
 *
 *              This routine performs the same image parsing and object-table
 *              reconstruction as the ordinary global-heap metadata-cache
 *              deserializer, but the resulting heap is not added to the
 *              file's CWFS list and is not registered as an independent
 *              metadata-cache object.
 *
 *              IMAGE is expected to contain only the embedded H5HG
 *              collection image. The structured-chunk layer must remove and
 *              validate any H5_SECTION_VL signature, implementation version,
 *              flags, checksum, or other section-level metadata before
 *              calling this routine.
 *
 *              The returned heap is owned by the decoded structured chunk
 *              and must eventually be released with H5HG__free_local().
 *
 * Return:      Success: Pointer to a newly decoded chunk-local heap
 *              Failure: NULL
 *
 *                                              -- AZO   7/14/26
 *
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__decode_local(H5F_t *f, const void *image, size_t len)
{
    H5HG_heap_t   *heap      = NULL; /* Local heap being reconstructed */
    uint8_t       *p         = NULL; /* Current record in copied heap image */
    const uint8_t *p_end     = NULL; /* Last valid byte in copied heap image */
    size_t         max_idx   = 0;    /* Largest nonzero object index observed */
    size_t         nalloc    = 0;    /* Initial object-table allocation size */
    size_t         need      = 0;    /* Complete serialized record size */
    unsigned       idx       = 0;    /* Decoded heap object index */
    uint8_t       *begin     = p;    /* Beginning of serialized object record */
    size_t         new_alloc = 0;    /* Enlarged table size */
    H5HG_obj_t    *new_obj   = NULL; /* Reallocated object table */
    size_t         aligned_size;     /* Aligned logical payload length */
    H5HG_heap_t   *ret_value = NULL; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(image);

    /*
     * Validate the length before calculating P_END. This prevents an
     * invalid or zero length from underflowing the end-pointer expression.
     */
    if ((len < (size_t)H5HG_MINSIZE))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                    "chunk-local H5HG image is smaller than the minimum heap size");

    /* Allocate the in-memory heap descriptor. */
    if ((NULL == (heap = H5FL_CALLOC(H5HG_heap_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap descriptor");

    /*
     * A local heap has no standalone file address. Retain the file-shared
     * state because decoding length fields depends on the file's configured
     * size width.
     */
    heap->addr   = HADDR_UNDEF;
    heap->shared = H5F_SHARED(f);

    /* Allocate and retain a private copy of the serialized collection. */
    if ((NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, len))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap image");

    H5MM_memcpy(heap->chunk, image, len);

    /*
     * Parse the copied image so all reconstructed object pointers refer to
     * HEAP->CHUNK rather than to the caller's input buffer.
     */
    p_end = heap->chunk + len - 1;

    /* Decode and validate the global-heap collection header. */
    if ((H5_IS_BUFFER_OVERFLOW(heap->chunk, H5HG_SIZEOF_HDR(f), p_end)))
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                    "ran off end of chunk-local heap image while decoding header");

    /*
     * Decode the embedded H5HG collection header locally. The equivalent
     * metadata-cache helper is static to H5HGcache.c, so the chunk-local path
     * performs the small amount of header parsing it needs here without
     * changing the legacy cache implementation.
     */
    {
        const uint8_t *hdr = heap->chunk;

        /* Collection signature */
        if (H5_IS_BUFFER_OVERFLOW(hdr, H5_SIZEOF_MAGIC, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding signature");

        if (memcmp(hdr, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC) != 0)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad chunk-local H5HG collection signature");

        hdr += H5_SIZEOF_MAGIC;

        /* Collection format version */
        if (H5_IS_BUFFER_OVERFLOW(hdr, 1, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding version");

        if (H5HG_VERSION != *hdr++)
            HGOTO_ERROR(H5E_HEAP, H5E_VERSION, NULL, "unsupported chunk-local H5HG collection version");

        /*
         * Skip the three reserved bytes in the existing H5HG collection
         * header. This preserves the legacy global-heap image format.
         */
        if (H5_IS_BUFFER_OVERFLOW(hdr, 3, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding reserved header bytes");

        hdr += 3;

        /* Complete serialized collection size */
        if (H5_IS_BUFFER_OVERFLOW(hdr, H5F_sizeof_size(f), p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding collection size");

        H5F_DECODE_LENGTH(f, hdr, heap->size);

        if (heap->size < H5HG_MINSIZE)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local H5HG collection size is too small");
    }

    /*
     * The supplied buffer must contain exactly one complete H5HG collection.
     * A mismatch indicates truncation or trailing bytes in the section data.
     */
    if ((heap->size != len))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                    "encoded heap size does not match chunk-local image length");

    /* Serialized object records begin after the aligned collection header. */
    p = heap->chunk + H5HG_SIZEOF_HDR(f);

    /*
     * Allocate a zero-initialized object table. H5HG records are not
     * required to appear in object-index order, so unused entries must begin
     * cleared.
     */
    nalloc = H5HG_NOBJS(f, heap->size);

    if ((0 == nalloc))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap object-table size");

    if ((NULL == (heap->obj = H5FL_SEQ_CALLOC(H5HG_obj_t, nalloc))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap object table");

    heap->nalloc = nalloc;
    /* Reconstruct the number of live payload objects while walking the serialized records. */
    heap->nlive  = 0;

    /*
     * Walk the serialized collection and reconstruct the in-memory object
     * table. Index zero represents free space; nonzero indices represent
     * stored payload objects.
     */
    while (p < (heap->chunk + heap->size)) {
        /*
         * A trailing extent smaller than an object header cannot contain a
         * payload record and is represented as headerless free space.
         */
        if (((p + H5HG_SIZEOF_OBJHDR(f)) > (heap->chunk + heap->size))) {
            if (NULL != heap->obj[0].begin)
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local free-space object is already defined");

            heap->obj[0].size  = (size_t)((heap->chunk + heap->size) - p);
            heap->obj[0].nrefs = 0;
            heap->obj[0].begin = p;

            /*
             * Move directly to the end of the collection. This extent was
             * calculated from the validated heap bounds.
             */
            p += heap->obj[0].size;
        }
        else {
            need  = 0; /* Complete serialized record size */
            idx   = 0; /* Decoded heap object index */
            begin = p; /* Beginning of serialized object record */

            /* Decode the two-byte object index. */
            if (H5_IS_BUFFER_OVERFLOW(p, 2, p_end))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding object index");

            UINT16DECODE(p, idx);

            /*
             * Enlarge the object table if the encoded object index exceeds
             * the initial size-based estimate.
             */
            if ((idx >= heap->nalloc)) {
                new_alloc = 0;    /* Enlarged table size */
                new_obj   = NULL; /* Reallocated object table */

                if ((heap->nalloc > (SIZE_MAX / 2)))
                    HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap object-table size overflow");

                new_alloc = MAX(heap->nalloc * 2, (size_t)idx + 1);

                if ((size_t)idx >= new_alloc)
                    HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap object index");

                if ((NULL == (new_obj = H5FL_SEQ_REALLOC(H5HG_obj_t, heap->obj, new_alloc))))
                    HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL,
                                "unable to enlarge chunk-local heap object table");

                memset(&new_obj[heap->nalloc], 0, (new_alloc - heap->nalloc) * sizeof(new_obj[0]));

                heap->obj    = new_obj;
                heap->nalloc = new_alloc;

                if ((heap->nalloc <= heap->nused))
                    HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                                "invalid chunk-local heap object-table allocation");
            }

            /*
             * Reject duplicate serialized object indices. Each index must
             * identify at most one record in the collection.
             */
            if ((heap->obj[idx].begin))
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "duplicate object index in chunk-local heap image");

            /* Decode the object's reference count. */
            if ((H5_IS_BUFFER_OVERFLOW(p, 2, p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding reference count");

            UINT16DECODE(p, heap->obj[idx].nrefs);

            /*
             * Skip the four reserved bytes in the existing H5HG object
             * header. Their format remains unchanged from normal global
             * heaps.
             */
            if ((H5_IS_BUFFER_OVERFLOW(p, 4, p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding reserved field");

            p += 4;

            /* Decode the logical, unaligned payload length. */
            if ((H5_IS_BUFFER_OVERFLOW(p, H5F_sizeof_size(f), p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding object size");

            H5F_DECODE_LENGTH(f, p, heap->obj[idx].size);

            /*
             * Object-table entries point to the beginning of the complete
             * serialized record, including the object header.
             */
            heap->obj[idx].begin = begin;

            if (idx > 0) {
                aligned_size = H5HG_ALIGN(heap->obj[idx].size);

                if (aligned_size < heap->obj[idx].size)
                    HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL,
                                "chunk-local heap object size overflow");

                if (aligned_size > SIZE_MAX - H5HG_SIZEOF_OBJHDR(f))
                    HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL,
                                "chunk-local heap object-record size overflow");

                /*
                * A normal object record contains the object header followed
                * by its aligned payload.
                */
                need = H5HG_SIZEOF_OBJHDR(f) + aligned_size;

                if ((size_t)idx > max_idx)
                    max_idx = (size_t)idx;
            }
            else {
                /*
                * Object zero's encoded size describes the complete
                * free-space extent, including its header.
                */
                need = heap->obj[0].size;
            }

            /* Verify that the complete record lies within the heap image. */
            if (H5_IS_BUFFER_OVERFLOW(begin, need, p_end))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "chunk-local heap object extends beyond heap image");

            /*
            * Each valid nonzero record represents one live payload object.
            * Valid zero-length payload objects are included.
            */
            if (idx > 0) {
                heap->nlive++;
            }

            p = begin + need;

        } /* end else */

    } /* end while */

    /* The parser must consume the complete encoded collection. */
    if (p != (heap->chunk + heap->size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local heap image was only partially decoded");

    if (!H5HG_ISALIGNED(heap->obj[0].size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "decoded chunk-local free-space extent is not aligned");

    /*
     * Set the next never-issued object index. Cleared entries below this
     * value can later be reused after the 16-bit index range wraps.
     */
    if (max_idx > 0)
        heap->nused = max_idx + 1;
    else
        heap->nused = 1;

    if (max_idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid next-unused chunk-local heap index");

    ret_value = heap;
    heap      = NULL;

done:
    /*
     * Local heaps have not been added to CWFS or H5AC, so partial decode
     * cleanup must use the chunk-local free routine.
     */
    if (heap)
        if (H5HG__free_local(heap) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL, "unable to destroy partially decoded chunk-local heap");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__decode_local() */