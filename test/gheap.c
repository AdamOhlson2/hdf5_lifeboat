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
 * Purpose:     Tests the global heap.  The global heap is the set of all
 *              collections but the collections are not related to one
 *              another by anything that appears in the file format.
 */
#include "h5test.h"
#include "H5ACprivate.h"
#include "H5CXprivate.h" /* API Contexts                         */
#include "H5Eprivate.h"
#include "H5Fprivate.h"
#include "H5Gprivate.h"
#include "H5HGprivate.h"
#include "H5Iprivate.h"
#include "H5Pprivate.h"
#include "H5VLprivate.h"

/* Macros for printing error messages in loops.  These print up to
 * GHEAP_REPEATED_ERR_LIM errors, and suppress the rest */
#define GHEAP_REPEATED_ERR_LIM 20

/* Number of heap objects to test */
#define GHEAP_TEST_NOBJS 1024

#define GHEAP_REPEATED_ERR(MSG)                                                                              \
    do {                                                                                                     \
        nerrors++;                                                                                           \
        if (nerrors <= GHEAP_REPEATED_ERR_LIM) {                                                             \
            H5_FAILED();                                                                                     \
            puts(MSG);                                                                                       \
            if (nerrors == GHEAP_REPEATED_ERR_LIM)                                                           \
                puts("    Suppressing further errors...");                                                   \
        }       /* end if */                                                                                 \
    } while (0) /* end GHEAP_REPEATED_ERR */

static const char *FILENAME[] = {"gheap1", "gheap2", "gheap3", "gheap4", "gheapooo",
                                 "lheap1", "lheap2", "lheap3", "lheap4", "lheapooo", "lheapencdec", NULL};

/*-------------------------------------------------------------------------
 * Function:    test_1
 *
 * Purpose:     Writes a sequence of objects to the global heap where each
 *              object is larger than the one before.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_1(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    uint8_t in[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("monotonically increasing lengths");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /*
     * Write the objects, monotonically increasing in length.  Since this is
     * a clean file, the addresses allocated for the collections should also
     * be monotonically increasing.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
        else if (u && H5_addr_gt(obj[u - 1].addr, obj[u].addr)) {
            H5_FAILED();
            puts("    Collection addresses are not monotonically increasing");
            nerrors++;
        }
    }

    /*
     * Now try to read each object back.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG_read(f, obj + u, in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Value read doesn't match value written");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_2
 *
 * Purpose:     Writes a sequence of objects to the global heap where each
 *              object is smaller than the one before.
 *
 * Return:      Success:    0
 *
 *              Failure:     number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_2(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    uint8_t in[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("monotonically decreasing lengths");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /*
     * Write the objects, monotonically decreasing in length.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (H5HG_insert(f, size, out, obj + u) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
    }

    /*
     * Now try to read each object back.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG_read(f, obj + u, in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Value read doesn't match value written");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_3
 *
 * Purpose:     Creates a few global heap objects and then removes them all.
 *              The collection should also be removed.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_3(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("complete object removal");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /* Create some stuff */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
    }

    /* Remove everything */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        status = H5HG_remove(f, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to remove object");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_4
 *
 * Purpose:     Tests the H5HG_remove() feature by writing lots of objects
 *              and occasionally removing some.  When we're done they're all
 *              removed.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_4(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("partial object removal");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        /* Insert */
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }

        /* Remove every third one beginning with the second, but after the
         * next one has already been inserted.  That is, insert A, B, C;
         * remove B, insert D, E, F; remove E; etc.
         */
        if (1 == (u % 3)) {
            H5Eclear2(H5E_DEFAULT);
            status = H5HG_remove(f, obj + u - 1);
            if (status < 0) {
                H5_FAILED();
                puts("    Unable to remove object");
                nerrors++;
            }
            memset(obj + u - 1, 0, sizeof *obj);
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_ooo_indices
 *
 * Purpose:     Tests that indices can be stored out of order.  This can
 *              happen when the indices "wrap around" due to many
 *              insertions and deletions (for example, from rewriting a
 *              VL dataset).
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_ooo_indices(hid_t fapl)
{
    hid_t    file = H5I_INVALID_HID;
    H5F_t   *f    = NULL;
    unsigned i, j;
    H5HG_t  *obj = NULL;
    herr_t   status;
    int      nerrors = 0;
    char     filename[1024];

    TESTING("out of order indices");

    if (NULL == (obj = (H5HG_t *)malloc(2000 * sizeof(*obj))))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[4], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /* Alternately insert 1000 entries and remove the previous group of 1000
     * entries, until the indices wrap around.
     */
    for (i = 0; i < 66; i++) {
        /* Insert 1000 entries.  The index into the obj array will alternate up
         * and down by 1000 so the previous set of insertions is preserved and
         * can be deleted.
         */
        for (j = 1000 * ((~i & 1)); j < 1000 * ((~i & 1) + 1); j++) {
            H5Eclear2(H5E_DEFAULT);
            status = H5HG_insert(f, sizeof(j), &j, &obj[j]);
            if (status < 0)
                GHEAP_REPEATED_ERR("    Unable to insert object into global heap");

            /* Check that the index is as expected */
            if (obj[j].idx != ((1000 * i) + j - (1000 * ((~i & 1)))) % ((1U << 16) - 1) + 1)
                GHEAP_REPEATED_ERR("    Unexpected global heap index");
        }

        /* Remove the previous 1000 entries */
        if (i > 0)
            for (j = 1000 * (i & 1); j < 1000 * ((i & 1) + 1); j++) {
                H5Eclear2(H5E_DEFAULT);
                status = H5HG_remove(f, &obj[j]);
                if (status < 0)
                    GHEAP_REPEATED_ERR("    Unable to remove object from global heap");
            }
    }

    /* The indices should have "wrapped around" on the last iteration */
    assert(obj[534].idx == 65535);
    assert(obj[535].idx == 1);

    /* Reopen the file */
    if (H5Fclose(file) < 0)
        goto error;
    if ((file = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to open file");
        goto error;
    } /* end if */

    /* Read the objects to make sure the heap is still readable */
    for (i = 0; i < 1000; i++) {
        if (NULL == H5HG_read(f, &obj[i], &j, NULL))
            goto error;
        if (i != j) {
            H5_FAILED();
            puts("    Incorrect read value");
            goto error;
        }
    }

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;
    free(obj);
    obj = NULL;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
} /* end test_ooo_indices */

/* Chunk-local H5HG tests adapted from the existing global heap tests.
 */

/*-------------------------------------------------------------------------
 * Function:    test_1_local
 *
 * Purpose:     Inserts objects whose sizes increase monotonically into one
 *              chunk-local heap and verifies that all payloads remain intact
 *              after repeated local-heap extensions.
 * 
 *                                           -- AZO   7/09/26
 *-------------------------------------------------------------------------
 */
static int
test_1_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];      /* Read-back buffer */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("local heap with monotonically increasing lengths");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /* Insert progressively larger objects into the same local heap. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, &heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
        else if (0 == obj_idx[u]) {
            H5_FAILED();
            puts("    Chunk-local heap returned reserved index zero");
            nerrors++;
        }
    }

    /* Verify all objects after the heap has grown repeatedly. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG__read_local(f, heap, obj_idx[u], in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object from chunk-local heap");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Local heap value read does not match value written");
            nerrors++;
        }
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;

    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_1_local() */

/*-------------------------------------------------------------------------
 * Function:    test_2_local
 *
 * Purpose:     Inserts objects whose sizes decrease monotonically into one
 *              chunk-local heap and verifies every payload.
 * 
 *                                        -- AZO   7/09/26
 *-------------------------------------------------------------------------
 */
static int
test_2_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];      /* Read-back buffer */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("local heap with monotonically decreasing lengths");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, &heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG__read_local(f, heap, obj_idx[u], in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object from chunk-local heap");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Local heap value read does not match value written");
            nerrors++;
        }
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_2_local() */

/*-------------------------------------------------------------------------
 * Function:    test_3_local
 *
 * Purpose:     Removes every payload from a chunk-local heap and confirms
 *              that no live object remains. The heap itself is explicitly
 *              freed because its lifetime belongs to the structured chunk.
 * 
 *                                          -- AZO   7/10/26
 *-------------------------------------------------------------------------
 */
static int
test_3_local(hid_t fapl)
{
    hid_t        file       = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f          = NULL;            /* Internal file object */
    H5HG_heap_t *heap       = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx    = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];        /* Object contents */
    size_t       u;                            /* Object index */
    size_t       size;                         /* Current payload size */
    hbool_t      heap_empty = false;           /* Removal empty-state result */
    htri_t       is_empty;                     /* Independent empty check */
    int          nerrors = 0;                  /* Number of failures */
    char         filename[1024];               /* Test file name */

    TESTING("complete chunk-local heap object removal");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, &heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
    }

    /* The heap must become empty only after the last live object is removed. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        heap_empty = false;

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__remove_local(f, heap, obj_idx[u], &heap_empty) < 0) {
            H5_FAILED();
            puts("    Unable to remove object from chunk-local heap");
            nerrors++;
        }
        else if (u + 1 < GHEAP_TEST_NOBJS && heap_empty) {
            H5_FAILED();
            puts("    Chunk-local heap reported empty before final removal");
            nerrors++;
        }
    }

    is_empty = H5HG__is_empty_local(heap);
    if (is_empty < 0) {
        H5_FAILED();
        puts("    Unable to determine whether chunk-local heap is empty");
        nerrors++;
    }
    else if (!heap_empty || !is_empty) {
        H5_FAILED();
        puts("    Chunk-local heap is not empty after all removals");
        nerrors++;
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_3_local() */

/*-------------------------------------------------------------------------
 * Function:    test_4_local
 *
 * Purpose:     Interleaves insertion and removal to exercise repeated local
 *              heap compaction. Surviving objects are verified before they
 *              are removed, and the heap must be empty at the end.
 * 
 * 
 *                                                -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_4_local(hid_t fapl)
{
    hid_t        file       = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f          = NULL;            /* Internal file object */
    H5HG_heap_t *heap       = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx    = NULL;            /* Local object indices */
    bool        *live       = NULL;            /* Whether each object is live */
    uint8_t      out[GHEAP_TEST_NOBJS];        /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];         /* Read-back buffer */
    size_t       u;                            /* Object index */
    size_t       size;                         /* Current payload size */
    hbool_t      heap_empty = false;           /* Final removal result */
    htri_t       is_empty;                     /* Independent empty check */
    int          nerrors = 0;                  /* Number of failures */
    char         filename[1024];               /* Test file name */

    TESTING("partial chunk-local heap object removal");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;
    if (NULL == (live = (bool *)calloc(GHEAP_TEST_NOBJS, sizeof(*live))))
        goto error;

    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, &heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
        else
            live[u] = true;

        /* Match the original test's interleaved removal pattern. */
        if (1 == (u % 3)) {
            H5Eclear2(H5E_DEFAULT);
            if (H5HG__remove_local(f, heap, obj_idx[u - 1], NULL) < 0) {
                H5_FAILED();
                puts("    Unable to remove object from chunk-local heap");
                nerrors++;
            }
            else
                live[u - 1] = false;
        }
    }

    /* Compaction must not alter any object that remains live. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        if (!live[u])
            continue;

        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG__read_local(f, heap, obj_idx[u], in, NULL)) {
            H5_FAILED();
            puts("    Unable to read surviving chunk-local heap object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Surviving chunk-local heap object was corrupted");
            nerrors++;
        }
    }

    /* Remove all surviving objects and verify the final empty state. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        if (!live[u])
            continue;

        heap_empty = false;
        H5Eclear2(H5E_DEFAULT);
        if (H5HG__remove_local(f, heap, obj_idx[u], &heap_empty) < 0) {
            H5_FAILED();
            puts("    Unable to remove surviving chunk-local heap object");
            nerrors++;
        }
        else
            live[u] = false;
    }

    is_empty = H5HG__is_empty_local(heap);
    if (is_empty < 0) {
        H5_FAILED();
        puts("    Unable to determine whether chunk-local heap is empty");
        nerrors++;
    }
    else if (!heap_empty || !is_empty) {
        H5_FAILED();
        puts("    Chunk-local heap is not empty after final removals");
        nerrors++;
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;
    free(live);
    live = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    free(live);
    return MAX(1, nerrors);
} /* end test_4_local() */

/*-------------------------------------------------------------------------
 * Function:    test_ooo_indices_local
 *
 * Purpose:     Forces the 16-bit local object index to wrap and reuse cleared
 *              entries. The heap is then encoded and decoded to verify that
 *              out-of-order indices survive the image round trip.
 * 
 *                                          -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_ooo_indices_local(hid_t fapl)
{
    hid_t        file      = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f         = NULL;            /* Internal file object */
    H5HG_heap_t *heap      = NULL;            /* Original local heap */
    H5HG_heap_t *decoded   = NULL;            /* Decoded local heap */
    size_t      *obj_idx   = NULL;            /* Current object indices */
    uint8_t     *image     = NULL;            /* Encoded heap image */
    size_t       image_len = 0;               /* Encoded image size */
    unsigned     i;                           /* Iteration group */
    unsigned     j;                           /* Object value/index */
    unsigned     value;                       /* Read-back value */
    size_t       expected;                    /* Expected local index */
    int          nerrors = 0;                 /* Number of failures */
    char         filename[1024];              /* Test file name */

    TESTING("chunk-local heap out-of-order indices");

    if (NULL == (obj_idx = (size_t *)calloc(2000, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[4], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /* Alternate two groups until never-used 16-bit indices are exhausted. */
    for (i = 0; i < 66; i++) {
        for (j = 1000 * ((~i & 1)); j < 1000 * ((~i & 1) + 1); j++) {
            H5Eclear2(H5E_DEFAULT);
            if (H5HG__insert_local(f, &heap, sizeof(j), &j, &obj_idx[j]) < 0)
                GHEAP_REPEATED_ERR("    Unable to insert object into chunk-local heap");

            expected = ((1000 * i) + j - (1000 * ((~i & 1)))) % ((1U << 16) - 1) + 1;
            if (obj_idx[j] != expected)
                GHEAP_REPEATED_ERR("    Unexpected chunk-local heap object index");
        }

        if (i > 0)
            for (j = 1000 * (i & 1); j < 1000 * ((i & 1) + 1); j++) {
                H5Eclear2(H5E_DEFAULT);
                if (H5HG__remove_local(f, heap, obj_idx[j], NULL) < 0)
                    GHEAP_REPEATED_ERR("    Unable to remove object from chunk-local heap");
            }
    }

    assert(obj_idx[534] == 65535);
    assert(obj_idx[535] == 1);

    /*
    * Export the complete local heap image without accessing the opaque
    * H5HG_heap_t representation directly.
    */
    if (H5HG__encode_local(heap, &image, &image_len) < 0)
        goto error;

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;

    if (NULL == (decoded = H5HG__decode_local(f, image, image_len)))
        goto error;

    /* The final live group occupies obj_idx[0..999]. */
    for (i = 0; i < 1000; i++) {
        value = 0;
        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG__read_local(f, decoded, obj_idx[i], &value, NULL)) {
            H5_FAILED();
            puts("    Unable to read decoded chunk-local heap object");
            nerrors++;
            break;
        }
        if (i != value) {
            H5_FAILED();
            puts("    Incorrect value read from decoded chunk-local heap");
            nerrors++;
            break;
        }
    }

    image = H5MM_xfree(image);
    if (H5HG__free_local(decoded) < 0)
        goto error;
    decoded = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    if (decoded)
        H5HG__free_local(decoded);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    image = H5MM_xfree(image);
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_OOO_indices_local() */

/*-------------------------------------------------------------------------
 * Function:    test_encode_decode_local
 *
 * Purpose:     Verifies that a chunk-local heap can be converted to its
 *              serialized H5HG image and reconstructed as a fully usable
 *              in-memory heap.
 *
 *              The test performs the following sequence:
 *
 *                  1. Create a local heap containing four payload objects.
 *
 *                  2. Remove one object before encoding. This leaves a
 *                     cleared object-table entry and forces the encoded
 *                     image to represent a heap whose indices are not all
 *                     live.
 *
 *                  3. Encode the heap into an independent byte buffer and
 *                     destroy the original H5HG_heap_t. This ensures the
 *                     decoder cannot rely on pointers or memory owned by
 *                     the original heap.
 *
 *                  4. Decode a new heap from the byte buffer and verify that
 *                     every object that was live at encode time can still be
 *                     found by its original local index and has unchanged
 *                     contents.
 *
 *                  5. Verify that the removed object remains absent after
 *                     decoding.
 *
 *                  6. Insert a new object into the decoded heap and read it
 *                     back. This verifies that decode reconstructed mutable
 *                     heap state, including the free-space record, object
 *                     table, object pointers, nused, and nalloc.
 *
 *                  7. Encode and decode the modified heap a second time and
 *                     verify that the newly inserted object survives that
 *                     round trip.
 *
 *              The second round trip is important because it verifies that
 *              a heap produced by H5HG__decode_local() can subsequently be
 *              modified and re-serialized, rather than being usable only
 *              for read-only access.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                          -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_encode_decode_local(hid_t fapl)
{
    /* 
     * Initial payloads stored in the local heap. The second payload is 
     * deliberately removed before encoding
     */
    static const char *values[] = {
        "first local payload",
        "payload removed before encoding",
        "third payload is deliberately longer than the first",
        "fourth"
    };

    hid_t        file        = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f           = NULL;            /* Internal file object */
    H5HG_heap_t *heap        = NULL;            /* Original local heap */
    H5HG_heap_t *decoded     = NULL;            /* First decoded heap */
    H5HG_heap_t *decoded_2   = NULL;            /* Second decoded heap */
    size_t       obj_idx[4]  = {0, 0, 0, 0};    /* Original object indices */
    size_t       new_idx     = 0;               /* Post-decode object index */
    uint8_t     *image       = NULL;            /* First encoded image */
    uint8_t     *image_2     = NULL;            /* Second encoded image */
    size_t       image_len   = 0;               /* First image size */
    size_t       image_len_2 = 0;               /* Second image size */
    char         read_buf[128];                 /* Payload read-back buffer */
    const char  *new_value = "inserted after decode";
    size_t       u;                             /* Value index */
    bool         removed_reappeared = false;    /* Removed-object check */
    int          nerrors = 0;                   /* Number of failures */
    char         filename[1024];                /* Test file name */

    TESTING("chunk-local heap encode/decode round trip");

    /* 
     * Create a file so the local H5HG routines can use the file's configured
     * encoded length width. The local heap itself is not written as a normal
     * file-backed global heap.
     */
    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);

    if ( (file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0 )
        goto error;

    if ( NULL == (f = (H5F_t *)H5VL_object(file)) ) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /*
     * Build the original local heap. Each returned index is retained because the same
     * index must identify the same surviving payload after decode. 
     */
    for (u = 0; u < 4; u++) {
        H5Eclear2(H5E_DEFAULT);

        if (H5HG__insert_local(f, &heap, (strlen(values[u]) + 1), values[u], &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object before local heap encoding");
            nerrors++;
        }
    }

    /*
     * Remove the second object before encoding. This creates a cleared 
     * object-table slot and tests whether the serialized image preserves
     * the distinction between live and removed indices.
     */
    if (H5HG__remove_local(f, heap, obj_idx[1], NULL) < 0) {
        H5_FAILED();
        puts("    Unable to remove object before local heap encoding");
        nerrors++;
        goto error;
    }

    /* 
     * Export the complete serialized H5HG collection image. The encoder
     * allocates IMAGE and retuns its exact IMAGE_LEN. 
     */
    if (H5HG__encode_local(heap, &image, &image_len) < 0) {
        H5_FAILED();
        puts("    Unable to encode chunk-local heap");
        nerrors++;
        goto error;
    }

    /* Destroy the source heap so decode cannot rely on its storage. */
    if (H5HG__free_local(heap) < 0)
        goto error;

    heap = NULL;

    /* Reconstruct a new in-memory heap from the serialized image */
    if (NULL == (decoded = H5HG__decode_local(f, image, image_len))) {
        H5_FAILED();
        puts("    Unable to decode chunk-local heap");
        nerrors++;
        goto error;
    }

    /* 
     * Verify all objects that were live when the heap was encoded.
     * Their original local indices and logical payload contents
     * must be preserved.
     */
    for (u = 0; u < 4; u++) {
        if (1 == u)
            continue;

        memset(read_buf, 0, sizeof(read_buf));
        H5Eclear2(H5E_DEFAULT);

        if (NULL == H5HG__read_local(f, decoded, obj_idx[u], read_buf, NULL)) {
            H5_FAILED();
            puts("    Unable to read surviving object after local heap decode");
            nerrors++;
        }
        else if (strcmp(read_buf, values[u]) != 0) {
            H5_FAILED();
            puts("    Decoded payload does not match encoded payload");
            nerrors++;
        }
    }

    /* 
     * Reading the removed index should fail. Supress the expected HDF5 error
     * stack while checking that the decoder did not recreate a cleared
     * object-table entry.
     */
    H5E_BEGIN_TRY
    {
        memset(read_buf, 0, sizeof(read_buf));
        removed_reappeared =
            (NULL != H5HG__read_local(f, decoded, obj_idx[1], read_buf, NULL));
    }
    H5E_END_TRY

    if ( removed_reappeared ) {
        H5_FAILED();
        puts("    Removed local heap object reappeared after decode");
        nerrors++;
    }

    /*
     * A correctly decoded heap must support normal mutation. Insert a new 
     * payload to verify that the decoder restored the free-space record,
     * object-table allocation, next-index state, and internal pointers. 
     */
    if (H5HG__insert_local(f, &decoded, strlen(new_value) + 1, new_value, &new_idx) < 0) {
        H5_FAILED();
        puts("    Unable to insert object into decoded chunk-local heap");
        nerrors++;
        goto error;
    }

    /* Verify the newly inserted payload immediately */
    memset(read_buf, 0, sizeof(read_buf));

    if (NULL == H5HG__read_local(f, decoded, new_idx, read_buf, NULL)) {
        H5_FAILED();
        puts("    Unable to read object inserted after decode");
        nerrors++;
    }
    else if (strcmp(read_buf, new_value) != 0) {
        H5_FAILED();
        puts("    Object inserted after decode has incorrect contents");
        nerrors++;
    }

    /* 
     * Serialize the modified decoded heap. This verifies that a heap
     * created by H5HG__decode_local() remains compatible with the
     * normal local encode path after subsequent mutation.
     */
    if (H5HG__encode_local(decoded, &image_2, &image_len_2) < 0) {
        H5_FAILED();
        puts("    Unable to re-encode modified local heap");
        nerrors++;
        goto error;
    }

    /*
     * Destroy the first decoded heap so the second decode must once again
     * reconstruct all in-memory state solely from the serialized image. 
     */
    if (H5HG__free_local(decoded) < 0)
        goto error;

    decoded = NULL;

    if (NULL == (decoded_2 = H5HG__decode_local(f, image_2, image_len_2))) {
        H5_FAILED();
        puts("    Unable to decode modified local heap image");
        nerrors++;
        goto error;
    }

    /* 
     * Confirm that the payload inserted after the first decode survived the second
     * encode/decode cycle at the same local index.
     */
    memset(read_buf, 0, sizeof(read_buf));

    if (NULL == H5HG__read_local(f, decoded_2, new_idx, read_buf, NULL)) {
        H5_FAILED();
        puts("    Unable to read post-decode insertion after second round trip");
        nerrors++;
    }
    else if (strcmp(read_buf, new_value) != 0) {
        H5_FAILED();
        puts("    Post-decode insertion did not survive second round trip");
        nerrors++;
    }

    /* Release both serialized images returned by H5HG__encode_local() */
    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    if (H5HG__free_local(decoded_2) < 0)
        goto error;

    decoded_2 = NULL;

    if (H5Fclose(file) < 0)
        goto error;

    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    /*
     * Release whichever heap representations were successfully created 
     * before the faulure. 
     */
    if (heap)
        H5HG__free_local(heap);

    if (decoded)
        H5HG__free_local(decoded);
        
    if (decoded_2)
        H5HG__free_local(decoded_2);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    /* H5HG__encode_local() allocates these images with H5MM_malloc() */
    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:	main
 *
 * Purpose:     Tests global heap.
 *
 * Return:      EXIT_SUCCESS/EXIT_FAILURE
 *
 *-------------------------------------------------------------------------
 */
int
main(void)
{
    int         nerrors        = 0;
    hid_t       fapl_id        = H5I_INVALID_HID;
    H5CX_node_t api_ctx        = {{0}, NULL}; /* API context node to push */
    bool        api_ctx_pushed = false;       /* Whether API context pushed */

    h5_test_init();
    if ((fapl_id = h5_fileaccess()) < 0)
        goto error;

    /* Push API context */
    if (H5CX_push(&api_ctx) < 0)
        FAIL_STACK_ERROR;
    api_ctx_pushed = true;

    nerrors += test_1(fapl_id);
    nerrors += test_2(fapl_id);
    nerrors += test_3(fapl_id);
    nerrors += test_4(fapl_id);
    nerrors += test_ooo_indices(fapl_id);

    if (nerrors)
        goto error;

    puts("All global heap tests passed.");

    /* Tests for the chunk-local heap */
    nerrors += test_1_local(fapl_id);
    nerrors += test_2_local(fapl_id);
    nerrors += test_3_local(fapl_id);
    nerrors += test_4_local(fapl_id);
    nerrors += test_ooo_indices_local(fapl_id);
    nerrors += test_encode_decode_local(fapl_id);

    /* Verify symbol table messages are cached */
    nerrors += (h5_verify_cached_stabs(FILENAME, fapl_id) < 0 ? 1 : 0);

    if (nerrors)
        goto error;

    puts("All chunk-local heap tests passed.");

    /* Pop API context */
    if (api_ctx_pushed && H5CX_pop(false) < 0)
        FAIL_STACK_ERROR;
    api_ctx_pushed = false;

    h5_cleanup(FILENAME, fapl_id);
    exit(EXIT_SUCCESS);

error:
    H5E_BEGIN_TRY
    {
        H5Pclose(fapl_id);
    }
    H5E_END_TRY

    if (api_ctx_pushed)
        H5CX_pop(false);

    puts("*** TESTS FAILED ***");
    exit(EXIT_FAILURE);
} /* end main() */
