#include "CudaKeyPoolDevice.h"
#include "Logger.h"
#include "util.h"
#include "AddressUtil.h"

// External CUDA functions (defined in CudaKeyPoolDevice.cu)
extern void callKeyPoolKernel(int blocks, int threads, int pointsPerThread, int compression);
extern cudaError_t initGTable(unsigned int *gxTable, unsigned int *gyTable);
extern cudaError_t setPoolDevicePointers(unsigned int *devX, unsigned int *devY, unsigned int *devPrivate, unsigned int *devChain);
extern cudaError_t allocatePoolChainBuf(unsigned int count);

void CudaKeyPoolDevice::cudaCall(cudaError_t err)
{
    if(err) {
        std::string errStr = cudaGetErrorString(err);
        throw KeySearchException(errStr);
    }
}

CudaKeyPoolDevice::CudaKeyPoolDevice(int device, int threads, int pointsPerThread, int blocks)
{
    cuda::CudaDeviceInfo info;
    try {
        info = cuda::getDeviceInfo(device);
        _deviceName = info.name;
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    if(threads <= 0 || threads % 32 != 0) {
        throw KeySearchException("The number of threads must be a multiple of 32");
    }

    if(pointsPerThread <= 0) {
        throw KeySearchException("At least 1 point per thread required");
    }

    if(blocks == 0) {
        if(threads % info.mpCount != 0) {
            throw KeySearchException("The number of threads must be a multiple of " + util::format("%d", info.mpCount));
        }

        _threads = threads / info.mpCount;
        _blocks = info.mpCount;

        while(_threads > 512) {
            _threads /= 2;
            _blocks *= 2;
        }
    } else {
        _threads = threads;
        _blocks = blocks;
    }

    _iterations = 0;
    _device = device;
    _pointsPerThread = pointsPerThread;
    _currentPoolIndex = 0;
    _totalPoolKeys = 0;

    _devPrivateKeys = NULL;
    _devX = NULL;
    _devY = NULL;
    _devGxTable = NULL;
    _devGyTable = NULL;
    _devChain = NULL;
}

CudaKeyPoolDevice::~CudaKeyPoolDevice()
{
    if(_devPrivateKeys) cudaFree(_devPrivateKeys);
    if(_devX) cudaFree(_devX);
    if(_devY) cudaFree(_devY);
    if(_devGxTable) cudaFree(_devGxTable);
    if(_devGyTable) cudaFree(_devGyTable);
    if(_devChain) cudaFree(_devChain);
}

void CudaKeyPoolDevice::initializeGTable()
{
    // Generate precomputed table: G, 2G, 4G, 8G, ..., 2^255*G
    std::vector<secp256k1::ecpoint> table;
    table.push_back(secp256k1::G());

    for(int i = 1; i < 256; i++) {
        secp256k1::ecpoint p = secp256k1::doublePoint(table[i - 1]);
        table.push_back(p);
    }

    // Allocate device memory for table
    cudaCall(cudaMalloc(&_devGxTable, 256 * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devGyTable, 256 * 8 * sizeof(unsigned int)));

    // Copy table to host buffer
    unsigned int *hostGx = new unsigned int[256 * 8];
    unsigned int *hostGy = new unsigned int[256 * 8];

    for(int i = 0; i < 256; i++) {
        unsigned int xWords[8], yWords[8];
        table[i].x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
        table[i].y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

        for(int j = 0; j < 8; j++) {
            hostGx[i * 8 + j] = xWords[j];
            hostGy[i * 8 + j] = yWords[j];
        }
    }

    // Copy to device
    cudaCall(cudaMemcpy(_devGxTable, hostGx, 256 * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_devGyTable, hostGy, 256 * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    delete[] hostGx;
    delete[] hostGy;

    // Initialize the constant memory pointers
    cudaCall(initGTable(_devGxTable, _devGyTable));
}

void CudaKeyPoolDevice::initWithPool(const std::vector<secp256k1::uint256> &keys, int compression)
{
    _keyPool = keys;
    _totalPoolKeys = keys.size();
    _currentPoolIndex = 0;
    _compression = compression;

    cudaCall(cudaSetDevice(_device));
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    size_t batchSize = (size_t)_blocks * _threads * _pointsPerThread;

    Logger::log(LogLevel::Info, "Initializing GPU for key pool mode");
    Logger::log(LogLevel::Info, "Batch size: " + util::formatThousands(batchSize) + " keys per iteration");
    Logger::log(LogLevel::Info, "Total pool keys: " + util::formatThousands(_totalPoolKeys));

    // Allocate device memory for batch processing
    cudaCall(cudaMalloc(&_devPrivateKeys, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devX, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devY, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devChain, batchSize * 8 * sizeof(unsigned int)));

    // Initialize G table for scalar multiplication
    initializeGTable();

    // Set device pointers in constant memory
    cudaCall(setPoolDevicePointers(_devX, _devY, _devPrivateKeys, _devChain));

    // Initialize result list
    cudaCall(_resultList.init(sizeof(CudaPoolDeviceResult), 16));

    Logger::log(LogLevel::Info, "GPU initialization complete");
}

void CudaKeyPoolDevice::init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride)
{
    // For pool mode, we don't use the standard init
    // Instead, use initWithPool()
    _compression = compression;

    cudaCall(cudaSetDevice(_device));
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    size_t batchSize = (size_t)_blocks * _threads * _pointsPerThread;

    // Allocate device memory
    cudaCall(cudaMalloc(&_devPrivateKeys, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devX, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devY, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMalloc(&_devChain, batchSize * 8 * sizeof(unsigned int)));

    // Initialize G table
    initializeGTable();

    // Set device pointers
    cudaCall(setPoolDevicePointers(_devX, _devY, _devPrivateKeys, _devChain));

    // Initialize result list
    cudaCall(_resultList.init(sizeof(CudaPoolDeviceResult), 16));
}

void CudaKeyPoolDevice::loadBatchToDevice(size_t startIdx, size_t count)
{
    size_t batchSize = (size_t)_blocks * _threads * _pointsPerThread;

    // Prepare host buffer for private keys
    unsigned int *hostKeys = new unsigned int[batchSize * 8];
    memset(hostKeys, 0, batchSize * 8 * sizeof(unsigned int));

    int totalThreads = _blocks * _threads;

    for(size_t i = 0; i < count && (startIdx + i) < _totalPoolKeys; i++) {
        // Calculate the strided position for this key
        int idx = i / totalThreads;           // Point index
        int threadId = i % totalThreads;      // Thread ID

        int base = idx * totalThreads * 8;
        int offset = threadId;

        unsigned int keyWords[8];
        _keyPool[startIdx + i].exportWords(keyWords, 8, secp256k1::uint256::BigEndian);

        // Store in strided format for coalesced memory access
        for(int k = 0; k < 8; k++) {
            hostKeys[base + offset + k * totalThreads] = keyWords[k];
        }
    }

    // Copy to device
    cudaCall(cudaMemcpy(_devPrivateKeys, hostKeys, batchSize * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Clear the X buffer (set to infinity to indicate uninitialized)
    cudaCall(cudaMemset(_devX, 0xFF, batchSize * 8 * sizeof(unsigned int)));
    cudaCall(cudaMemset(_devY, 0xFF, batchSize * 8 * sizeof(unsigned int)));

    delete[] hostKeys;
}

void CudaKeyPoolDevice::doStep()
{
    size_t batchSize = (size_t)_blocks * _threads * _pointsPerThread;

    // Check if we have more keys to process
    if(_currentPoolIndex >= _totalPoolKeys) {
        return;
    }

    // Calculate how many keys to process in this batch
    size_t keysToProcess = std::min(batchSize, _totalPoolKeys - _currentPoolIndex);

    // Load batch to device
    loadBatchToDevice(_currentPoolIndex, keysToProcess);

    // Execute kernel - performs scalar multiplication and hash checking
    try {
        callKeyPoolKernel(_blocks, _threads, _pointsPerThread, _compression);
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    // Get results
    getResultsInternal();

    // Move to next batch
    _currentPoolIndex += batchSize;
    _iterations++;
}

void CudaKeyPoolDevice::setTargets(const std::set<KeySearchTarget> &targets)
{
    _targets.clear();

    for(std::set<KeySearchTarget>::iterator i = targets.begin(); i != targets.end(); ++i) {
        hash160 h(i->value);
        _targets.push_back(h);
    }

    cudaCall(_targetLookup.setTargets(_targets));
}

bool CudaKeyPoolDevice::isTargetInList(const unsigned int hash[5])
{
    size_t count = _targets.size();

    while(count) {
        if(memcmp(hash, _targets[count - 1].h, 20) == 0) {
            return true;
        }
        count--;
    }

    return false;
}

void CudaKeyPoolDevice::removeTargetFromList(const unsigned int hash[5])
{
    size_t count = _targets.size();

    while(count) {
        if(memcmp(hash, _targets[count - 1].h, 20) == 0) {
            _targets.erase(_targets.begin() + count - 1);
            return;
        }
        count--;
    }
}

void CudaKeyPoolDevice::getResultsInternal()
{
    int count = _resultList.size();
    if(count == 0) {
        return;
    }

    unsigned char *ptr = new unsigned char[count * sizeof(CudaPoolDeviceResult)];
    _resultList.read(ptr, count);

    int actualCount = 0;

    for(int i = 0; i < count; i++) {
        struct CudaPoolDeviceResult *rPtr = &((struct CudaPoolDeviceResult *)ptr)[i];

        if(!isTargetInList(rPtr->digest)) {
            continue;
        }
        actualCount++;

        KeySearchResult minerResult;

        // Get the private key directly from the result (stored during kernel execution)
        minerResult.privateKey = secp256k1::uint256(rPtr->privateKey, secp256k1::uint256::BigEndian);
        minerResult.compressed = rPtr->compressed;

        memcpy(minerResult.hash, rPtr->digest, 20);

        minerResult.publicKey = secp256k1::ecpoint(
            secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian),
            secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian)
        );

        removeTargetFromList(rPtr->digest);
        _results.push_back(minerResult);
    }

    delete[] ptr;
    _resultList.clear();

    if(actualCount) {
        cudaCall(_targetLookup.setTargets(_targets));
    }
}

bool CudaKeyPoolDevice::verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed)
{
    secp256k1::ecpoint g = secp256k1::G();
    secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

    if(!(p == publicKey)) {
        return false;
    }

    unsigned int xWords[8];
    unsigned int yWords[8];

    p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    unsigned int digest[5];
    if(compressed) {
        Hash::hashPublicKeyCompressed(xWords, yWords, digest);
    } else {
        Hash::hashPublicKey(xWords, yWords, digest);
    }

    for(int i = 0; i < 5; i++) {
        if(digest[i] != hash[i]) {
            return false;
        }
    }

    return true;
}

size_t CudaKeyPoolDevice::getResults(std::vector<KeySearchResult> &resultsOut)
{
    for(size_t i = 0; i < _results.size(); i++) {
        resultsOut.push_back(_results[i]);
    }
    _results.clear();

    return resultsOut.size();
}

uint64_t CudaKeyPoolDevice::keysPerStep()
{
    return (uint64_t)_blocks * _threads * _pointsPerThread;
}

std::string CudaKeyPoolDevice::getDeviceName()
{
    return _deviceName;
}

void CudaKeyPoolDevice::getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem)
{
    cudaCall(cudaMemGetInfo(&freeMem, &totalMem));
}

secp256k1::uint256 CudaKeyPoolDevice::getNextKey()
{
    if(_currentPoolIndex < _totalPoolKeys) {
        return _keyPool[_currentPoolIndex];
    }
    return secp256k1::uint256(0);
}

bool CudaKeyPoolDevice::isPoolExhausted() const
{
    return _currentPoolIndex >= _totalPoolKeys;
}

size_t CudaKeyPoolDevice::getProcessedKeys() const
{
    return std::min(_currentPoolIndex, _totalPoolKeys);
}

size_t CudaKeyPoolDevice::getTotalPoolKeys() const
{
    return _totalPoolKeys;
}
