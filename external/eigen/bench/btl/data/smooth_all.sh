#! /bin/bash
ORIG_DIR=$1
SMOOTH_DIR=${ORIG_DIR}_smooth
mkdir ${SMOOTH_DIR}

AXPY_FILE=`find ${ORIG_DIR} -name "*.dat" | grep axpy`
for FILE in ${AXPY_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    ./smooth ${ORIG_DIR}/${BASE} 4 ${SMOOTH_DIR}/${BASE}_tmp
    ./regularize ${SMOOTH_DIR}/${BASE}_tmp 2500 15000 ${SMOOTH_DIR}/${BASE}
    rm -f  ${SMOOTH_DIR}/${BASE}_tmp
done


MATRIX_VECTOR_FILE=`find ${ORIG_DIR} -name "*.dat" | grep matrix_vector`
for FILE in ${MATRIX_VECTOR_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    ./smooth ${ORIG_DIR}/${BASE} 4 ${SMOOTH_DIR}/${BASE}_tmp
    ./regularize ${SMOOTH_DIR}/${BASE}_tmp 50 180 ${SMOOTH_DIR}/${BASE}
    rm -f  ${SMOOTH_DIR}/${BASE}_tmp
done

MATRIX_MATRIX_FILE=`find ${ORIG_DIR} -name "*.dat" | grep matrix_matrix`
for FILE in ${MATRIX_MATRIX_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    ./smooth ${ORIG_DIR}/${BASE} 4 ${SMOOTH_DIR}/${BASE}
done

AAT_FILE=`find ${ORIG_DIR} -name "*.dat" | grep _aat`
for FILE in ${AAT_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    ./smooth ${ORIG_DIR}/${BASE} 4 ${SMOOTH_DIR}/${BASE}
done


ATA_FILE=`find ${ORIG_DIR} -name "*.dat" | grep _ata`
for FILE in ${ATA_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    ./smooth ${ORIG_DIR}/${BASE} 4 ${SMOOTH_DIR}/${BASE}
done

### no smoothing for tinyvector and matrices libs

TINY_BLITZ_FILE=`find ${ORIG_DIR} -name "*.dat" | grep tiny_blitz`
for FILE in ${TINY_BLITZ_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    cp ${ORIG_DIR}/${BASE} ${SMOOTH_DIR}/${BASE}
done

TVMET_FILE=`find ${ORIG_DIR} -name "*.dat" | grep tvmet`
for FILE in ${TVMET_FILE}
do
    echo $FILE
    BASE=${FILE##*/}
    cp ${ORIG_DIR}/${BASE} ${SMOOTH_DIR}/${BASE}
done
