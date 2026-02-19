#!/usr/bin/env bash

## change this if ROCm is installed in a non-standard path
ROCM_PATH=/opt/rocm
MPI_INSTALL_DIR=/opt/mpich
RCCL_INSTALL_DIR=${ROCM_PATH}
 
## to use pre-installed MPI, change `build_mpi` to 0 and ensure that libmpi.so exists at `MPI_INSTALL_DIR/lib`.
build_mpi=1
## to use pre-installed RCCL, change `build_rccl` to 0 and ensure that librccl.so exists at `RCCL_INSTALL_DIR/lib`.
build_rccl=1
rccl_debug_mode=1
build_rccl_tests=1
mpi_mode=1
 
#Take the name from user use the following default
WORKDIR="$PWD/temp"
mkdir -p $WORKDIR
 
 
## building mpich
if [ ${build_mpi} -eq 1 ]
then
    cd ${WORKDIR}
    if [ ! -d mpich ]
    then
        wget https://www.mpich.org/static/downloads/4.1.2/mpich-4.1.2.tar.gz
        mkdir -p mpich
        tar -zxf mpich-4.1.2.tar.gz -C mpich --strip-components=1
        cd mpich
        mkdir build
        cd build
        ../configure --prefix=${WORKDIR}/mpich/install --disable-fortran --with-ucx=embedded
        make -j 16
        make install
    fi
    MPI_INSTALL_DIR=${WORKDIR}/mpich/install
fi
 
 
## building rccl
if [ ${build_rccl} -eq 1 ]
then
    RCCL_BUILD_TYPE=release
    cd ${WORKDIR}
    if [ ! -d rocm-systems/projects/rccl ]
    then
        git clone https://github.com/ROCm/rocm-systems.git
    fi
    cd ${WORKDIR}/rocm-systems/projects/rccl
    if [ ${rccl_debug_mode} -eq 1 ]
    then
        ./install.sh -l --debug --jobs $(nproc)
        RCCL_BUILD_TYPE=debug
        echo "Building rccl in debug mode"
    else
        ./install.sh -l --jobs $(nproc)
        RCCL_BUILD_TYPE=release
    fi
    RCCL_INSTALL_DIR=${WORKDIR}/rocm-systems/projects/rccl/build/${RCCL_BUILD_TYPE}
fi
 
 
## building rccl-tests (develop)
cd ${WORKDIR}
if [ ${build_rccl_tests} -eq 1 ]
then
  cd ${WORKDIR}/rocm-systems/projects/rccl-tests
  make clean
  make MPI=${mpi_mode} MPI_HOME=${MPI_INSTALL_DIR} NCCL_HOME=${RCCL_INSTALL_DIR} -j
fi

## running rccl-tests sweep
n_gpus=8                             # assuming 8 GPUs per node
n_nodes=1
total=$((n_gpus * n_nodes))                # total number of MPI ranks (1 per GPU)
echo "Total ranks: ${total}"    # print number of GPUs
cd ${WORKDIR}
# Get today's UTC date in yyyy_MM_dd format
DATE_UTC=$(date -u +"%Y_%m_%d")
# Set performance data directory name
PERF_DATA_DIR="perfdata_${DATE_UTC}"
mkdir -p ${PERF_DATA_DIR}

#Run parameters
b=1        #begin size
e=16G      #end size
d=float     #data types
n=1        #iterations
w=0        #warm up iterations
N=1        #stress cycle iterations
 
for coll in all_reduce #all_gather alltoall alltoallv broadcast gather reduce reduce_scatter scatter sendrecv
do
    # using MPICH; comment next line if using OMPI
    if [[ $mpi_mode -eq 1 ]]; then
        ${MPI_INSTALL_DIR}/bin/mpirun -np ${total} --bind-to numa -env NCCL_DEBUG=VERSION -env PATH=${MPI_INSTALL_DIR}/bin:${ROCM_PATH}/bin:$PATH -env LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}:${MPI_INSTALL_DIR}/lib:$LD_LIBRARY_PATH ${WORKDIR}/rocm-systems/projects/rccl-tests/build/${coll}_perf -b ${b} -e ${e} -f 2 -g 1 -d ${d} -n ${n} -w ${w} -N ${N} -M 1 2>&1 | tee ${WORKDIR}/${PERF_DATA_DIR}/${coll}.txt
    else 
         NCCL_DEBUG=VERSION PATH=${ROCM_PATH}/bin:$PATH LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}:$LD_LIBRARY_PATH ${WORKDIR}/rocm-systems/projects/rccl-tests/build/${coll}_perf -b ${b} -e ${e} -f 2 -g ${n_gpus} -d ${d} -n ${n} -w ${w} -N ${N} 2>&1 | tee ${WORKDIR}/${PERF_DATA_DIR}/${coll}.txt
    fi 
done