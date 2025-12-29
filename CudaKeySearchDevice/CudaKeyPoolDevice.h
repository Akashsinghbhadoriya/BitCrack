#ifndef _CUDA_KEY_POOL_DEVICE_H
#define _CUDA_KEY_POOL_DEVICE_H

#include "KeySearchDevice.h"
#include <vector>
#include <cuda_runtime.h>
#include "secp256k1.h"
#include "CudaDeviceKeys.h"
#include "CudaHashLookup.h"
#include "CudaAtomicList.h"
#include "cudaUtil.h"

// Structures that exist on both host and device side
struct CudaPoolDeviceResult {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
    unsigned int privateKey[8];  // Store the actual private key for pool mode
};

/**
 * CudaKeyPoolDevice - GPU device for processing keys from a custom pool
 *
 * Unlike CudaKeySearchDevice which generates sequential keys, this device
 * processes arbitrary keys from a pre-computed pool. Each iteration:
 * 1. Loads a batch of private keys from the pool
 * 2. Performs full scalar multiplication (k * G) on GPU
 * 3. Hashes and checks against targets
 *
 * This is slower than sequential mode (~20-50% of original speed) because:
 * - Full scalar multiplication instead of point addition
 * - CPU-to-GPU transfer for each batch
 * - No sequential memory access patterns
 */
class CudaKeyPoolDevice : public KeySearchDevice {

private:
    int _device;
    int _blocks;
    int _threads;
    int _pointsPerThread;
    int _compression;

    std::vector<KeySearchResult> _results;
    std::string _deviceName;

    uint64_t _iterations;
    size_t _currentPoolIndex;
    size_t _totalPoolKeys;

    // Pool of private keys to process
    std::vector<secp256k1::uint256> _keyPool;

    // Device memory for private keys (current batch)
    unsigned int *_devPrivateKeys;

    // Device memory for public keys (computed from private keys)
    unsigned int *_devX;
    unsigned int *_devY;

    // Precomputed table of G multiples for fast scalar multiplication
    unsigned int *_devGxTable;  // 256 entries: G, 2G, 4G, ..., 2^255*G
    unsigned int *_devGyTable;

    // Working memory for batch operations
    unsigned int *_devChain;

    CudaAtomicList _resultList;
    CudaHashLookup _targetLookup;

    std::vector<hash160> _targets;

    void cudaCall(cudaError_t err);
    void initializeGTable();
    void loadBatchToDevice(size_t startIdx, size_t count);
    void getResultsInternal();
    bool isTargetInList(const unsigned int hash[5]);
    void removeTargetFromList(const unsigned int hash[5]);

    bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed);

public:
    CudaKeyPoolDevice(int device, int threads, int pointsPerThread, int blocks = 0);
    ~CudaKeyPoolDevice();

    // Initialize with a pool of private keys
    void initWithPool(const std::vector<secp256k1::uint256> &keys, int compression);

    // Standard KeySearchDevice interface
    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride);

    virtual void doStep();

    virtual void setTargets(const std::set<KeySearchTarget> &targets);

    virtual size_t getResults(std::vector<KeySearchResult> &results);

    virtual uint64_t keysPerStep();

    virtual std::string getDeviceName();

    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem);

    virtual secp256k1::uint256 getNextKey();

    // Check if all pool keys have been processed
    bool isPoolExhausted() const;

    // Get progress
    size_t getProcessedKeys() const;
    size_t getTotalPoolKeys() const;
};

#endif
