## change this if ROCm is installed in a non-standard path
ROCM_PATH=/opt/rocm
 
## to use pre-installed MPI, change `build_mpi` to 0 and ensure that libmpi.so exists at `MPI_INSTALL_DIR/lib`.
build_mpi=0
MPI_INSTALL_DIR=/opt/mpich
 
## to use pre-installed RCCL, change `build_rccl` to 0 and ensure that librccl.so exists at `RCCL_INSTALL_DIR/lib`.
build_rccl=1
RCCL_INSTALL_DIR=${ROCM_PATH}
 
 #ls /usr
WORKDIR=$PWD/temp
 
 
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
 
 
## building rccl (develop)
if [ ${build_rccl} -eq 1 ]
then
    cd ${WORKDIR}
    if [ ! -d rccl ]
    then
        git clone https://github.com/ROCm/rccl -b develop
        cd rccl
        ./install.sh -l
    fi
    RCCL_INSTALL_DIR=${WORKDIR}/rccl/build/release
fi
 
 
## building rccl-tests (develop)
cd ${WORKDIR}
if [ ! -d rccl-tests ]
then
    git clone https://github.com/ROCm/rccl-tests 
fi
cd rccl-tests
make clean
make MPI=1 MPI_HOME=${MPI_INSTALL_DIR} NCCL_HOME=${RCCL_INSTALL_DIR} -j

## running rccl-tests sweep
m=8                             # assuming 8 GPUs per node
total=$((1 * m))                # total number of MPI ranks (1 per GPU)
echo "Total ranks: ${total}"    # print number of GPUs
cd ${WORKDIR}
mkdir perfdata
 
for coll in all_reduce all_gather alltoall alltoallv broadcast gather reduce reduce_scatter scatter sendrecv
do
    # using MPICH; comment next line if using OMPI
    ${MPI_INSTALL_DIR}/bin/mpirun -np ${total} --bind-to numa -env NCCL_DEBUG=VERSION -env PATH=${MPI_INSTALL_DIR}/bin:${ROCM_PATH}/bin:$PATH -env LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}/lib:${MPI_INSTALL_DIR}/lib:$LD_LIBRARY_PATH ${WORKDIR}/rccl-tests/build/${coll}_perf -b 1 -e 16G -f 2 -g 1 -d all -n 20 -w 5 -N 10 2>&1 | tee ${WORKDIR}/perfdata/${coll}.txt
done