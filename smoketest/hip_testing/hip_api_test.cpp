#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <cstring>

#define CHECK_HIP(cmd)                                                     \
  do {                                                                      \
    hipError_t e = cmd;                                                     \
    if (e != hipSuccess) {                                                  \
      std::cerr << "HIP error: " << hipGetErrorString(e)                    \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

void test_device_info() {
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    
    int driverVersion = 0, runtimeVersion = 0;
    CHECK_HIP(hipDriverGetVersion(&driverVersion));
    CHECK_HIP(hipRuntimeGetVersion(&runtimeVersion));

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        CHECK_HIP(hipGetDeviceProperties(&props, i));
        
        int busId = -1;
        CHECK_HIP(hipDeviceGetAttribute(&busId, hipDeviceAttributePciBusId, i));
        
        char pciBusIdStr[20];
        CHECK_HIP(hipDeviceGetPCIBusId(pciBusIdStr, 20, i));
        
        std::cout << "Device " << i << ": " << props.name << " (Bus ID: " << pciBusIdStr << ")" << std::endl;
    }
}

void test_memory_apis() {
    int currentDev;
    CHECK_HIP(hipGetDevice(&currentDev));
    int dev = 0;
    CHECK_HIP(hipSetDevice(dev));

    // Standard and Managed Malloc
    void *d_ptr, *m_ptr;
    CHECK_HIP(hipMalloc(&d_ptr, 1024));
    CHECK_HIP(hipMallocManaged(&m_ptr, 1024));

    // Memset and Address Range
    CHECK_HIP(hipMemset(d_ptr, 0, 1024));
    hipDeviceptr_t base;
    size_t size;
    CHECK_HIP(hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)d_ptr));

    // Peer Access Check (Requires 2+ GPUs)
    int count;
    CHECK_HIP(hipGetDeviceCount(&count));
    if (count > 1) {
        int canAccess = 0;
        CHECK_HIP(hipDeviceCanAccessPeer(&canAccess, 0, 1));
        std::cout << "P2P Access (0->1): " << (canAccess ? "Yes" : "No") << std::endl;
    }

    CHECK_HIP(hipFree(d_ptr));
    CHECK_HIP(hipFree(m_ptr));
}

void host_callback(void* data) {
    std::cout << "Host Node executed!" << std::endl;
}

void test_graphs() {
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphExec_t instance;

    CHECK_HIP(hipStreamCreate(&stream));

    // Allocate BEFORE capture
    void* d_ptr = nullptr;
    CHECK_HIP(hipMalloc(&d_ptr, 1024));

    // Begin capture
    CHECK_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    CHECK_HIP(hipMemsetAsync(d_ptr, 1, 1024, stream));

    hipError_t endStatus = hipStreamEndCapture(stream, &graph);
    if (endStatus != hipSuccess) {
        std::cerr << "Stream capture failed\n";
        CHECK_HIP(endStatus);
    }

    // Get captured nodes so we can add dependency
    size_t numNodes = 0;
    CHECK_HIP(hipGraphGetNodes(graph, nullptr, &numNodes));

    std::vector<hipGraphNode_t> nodes(numNodes);
    CHECK_HIP(hipGraphGetNodes(graph, nodes.data(), &numNodes));

    // Add Host Node AFTER captured nodes
    hipGraphNode_t hostNode;
    hipHostNodeParams hostParams{};
    hostParams.fn = host_callback;
    hostParams.userData = nullptr;

    CHECK_HIP(hipGraphAddHostNode(
        &hostNode,
        graph,
        nodes.data(),   // dependency
        numNodes,
        &hostParams));

    CHECK_HIP(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    CHECK_HIP(hipGraphLaunch(instance, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    // Cleanup
    CHECK_HIP(hipGraphExecDestroy(instance));
    CHECK_HIP(hipGraphDestroy(graph));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(d_ptr));
}




__global__ void test_bf16_kernel(hip_bfloat16* data) {
    float f = static_cast<float>(data[0]);  // convert bf16 → float
    f = f + 1.0f;
    data[0] = hip_bfloat16(f);              // convert float → bf16
}

void test_bfloat16() {
    hip_bfloat16 h_bf = hip_bfloat16(1.5f);
    hip_bfloat16* d_bf = nullptr;

    CHECK_HIP(hipMalloc(&d_bf, sizeof(hip_bfloat16)));
    CHECK_HIP(hipMemcpy(d_bf, &h_bf,
                        sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));

    test_bf16_kernel<<<1, 1>>>(d_bf);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(&h_bf, d_bf,
                        sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));

    std::cout << "Bfloat16 result: "
              << static_cast<float>(h_bf) << std::endl;

    CHECK_HIP(hipFree(d_bf));
}

void test_pointer_and_events() {
    float *d_ptr = nullptr;
    CHECK_HIP(hipMalloc(&d_ptr, sizeof(float)));

    // hipPointerGetAttribute
    hipPointerAttribute_t memoryType;
    CHECK_HIP(hipPointerGetAttribute(
        &memoryType,
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
        (void*)d_ptr));

    // Events
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreateWithFlags(&start, hipEventDefault));
    CHECK_HIP(hipEventCreateWithFlags(&stop, hipEventBlockingSync));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    CHECK_HIP(hipFree(d_ptr));
}
__global__ void simple_kernel(float* data) {
    *data = 1.0f;
}

void test_kernel_ext() {
    hipFuncAttributes attr;
    CHECK_HIP(hipFuncGetAttributes(
        &attr,
        (const void*)simple_kernel));

    float *d_ptr = nullptr;
    CHECK_HIP(hipMalloc(&d_ptr, sizeof(float)));
    dim3 grid(1), block(1);
    void* args[] = { &d_ptr };
    CHECK_HIP(hipExtLaunchKernel(
        (const void*)simple_kernel,
        grid,
        block,
        args,
        0,
        0,      // default stream
        nullptr,
        nullptr,
        0));

    CHECK_HIP(hipDeviceSynchronize());
    float h_out = 0.0f;
    CHECK_HIP(hipMemcpy(&h_out, d_ptr, sizeof(float), hipMemcpyDeviceToHost));
    if (h_out != 1.0f) {
        std::cerr << "Kernel result incorrect!\n";
        std::exit(EXIT_FAILURE);
    }
    CHECK_HIP(hipFree(d_ptr));
}

void test_async_and_stream_ops() {
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    float *d_ptr = nullptr;
    float h_val = 5.0f;

    CHECK_HIP(hipMalloc(&d_ptr, sizeof(float)));

    CHECK_HIP(hipMemcpyAsync(d_ptr, &h_val,
                             sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));

    hipEvent_t evt;
    CHECK_HIP(hipEventCreate(&evt));
    CHECK_HIP(hipEventRecord(evt, stream));

    CHECK_HIP(hipStreamWaitEvent(stream, evt, 0));

    hipError_t q = hipStreamQuery(stream);
    if (q != hipSuccess && q != hipErrorNotReady)
        CHECK_HIP(q);

    CHECK_HIP(hipStreamSynchronize(stream));

    CHECK_HIP(hipEventDestroy(evt));
    CHECK_HIP(hipFree(d_ptr));
    CHECK_HIP(hipStreamDestroy(stream));
}

void test_host_memory() {
    void* h_ptr = nullptr;

    CHECK_HIP(hipHostMalloc(&h_ptr, 1024));
    std::memset(h_ptr, 0, 1024);
    CHECK_HIP(hipHostFree(h_ptr));
}


#include <iostream>
#include <hip/hip_runtime.h>

// Forward declarations of the functions provided in the previous response
void test_device_info();
void test_memory_apis();
void test_graphs();
void test_bfloat16();
void test_pointer_and_events();
void test_kernel_ext();
void test_async_and_stream_ops();
void test_host_memory();

int main() {
    std::cout << "--- Starting HIP API Functional Test Suite ---\n";

    std::cout << "\n[1] Device & Runtime Info...\n";
    test_device_info();

    std::cout << "\n[2] Memory & Peer Access...\n";
    test_memory_apis();

    std::cout << "\n[3] Graphs & Stream Capture...\n";
    test_graphs();

    std::cout << "\n[4] Bfloat16...\n";
    test_bfloat16();

    std::cout << "\n[5] Pointer & Events...\n";
    test_pointer_and_events();

    std::cout << "\n[6] hipExtLaunchKernel...\n";
    test_kernel_ext();

    std::cout << "\n[7] Async & Stream Ops...\n";
    test_async_and_stream_ops();

    std::cout << "\n[8] Host Memory...\n";
    test_host_memory();

    std::cout << "\n--- All Tests Completed Successfully ---\n";

    CHECK_HIP(hipDeviceReset());
    return 0;
}
