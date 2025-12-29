#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeyPoolDevice.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"

// Constant memory for G table pointers
__constant__ unsigned int *_POOL_GX_TABLE[1];
__constant__ unsigned int *_POOL_GY_TABLE[1];

// Constant memory for device array pointers
__constant__ unsigned int *_POOL_X_PTR[1];
__constant__ unsigned int *_POOL_Y_PTR[1];
__constant__ unsigned int *_POOL_PRIVATE_PTR[1];
__constant__ unsigned int *_POOL_CHAIN[1];

// Initialize G table pointers in constant memory
cudaError_t initGTable(unsigned int *gxTable, unsigned int *gyTable)
{
    cudaError_t err = cudaMemcpyToSymbol(_POOL_GX_TABLE, &gxTable, sizeof(unsigned int *));
    if(err) return err;

    return cudaMemcpyToSymbol(_POOL_GY_TABLE, &gyTable, sizeof(unsigned int *));
}

// Set device pointers in constant memory
cudaError_t setPoolDevicePointers(unsigned int *devX, unsigned int *devY, unsigned int *devPrivate, unsigned int *devChain)
{
    cudaError_t err;

    err = cudaMemcpyToSymbol(_POOL_X_PTR, &devX, sizeof(unsigned int *));
    if(err) return err;

    err = cudaMemcpyToSymbol(_POOL_Y_PTR, &devY, sizeof(unsigned int *));
    if(err) return err;

    err = cudaMemcpyToSymbol(_POOL_PRIVATE_PTR, &devPrivate, sizeof(unsigned int *));
    if(err) return err;

    return cudaMemcpyToSymbol(_POOL_CHAIN, &devChain, sizeof(unsigned int *));
}

// Read a 256-bit integer from strided memory layout
__device__ void readPoolInt(const unsigned int *base, int idx, unsigned int *out)
{
    int totalThreads = gridDim.x * blockDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int start = idx * totalThreads * 8 + threadId;

    for(int i = 0; i < 8; i++) {
        out[i] = base[start + i * totalThreads];
    }
}

// Write a 256-bit integer to strided memory layout
__device__ void writePoolInt(unsigned int *base, int idx, const unsigned int *val)
{
    int totalThreads = gridDim.x * blockDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int start = idx * totalThreads * 8 + threadId;

    for(int i = 0; i < 8; i++) {
        base[start + i * totalThreads] = val[i];
    }
}

// Check if a point is at infinity (all 0xFF)
__device__ bool isPoolInfinity(const unsigned int *x)
{
    for(int i = 0; i < 8; i++) {
        if(x[i] != 0xFFFFFFFF) {
            return false;
        }
    }
    return true;
}

// Finalize RIPEMD160 hash
__device__ void doPoolRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}

// Hash public key (uncompressed)
__device__ void hashPoolPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

// Hash public key (compressed)
__device__ void hashPoolPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

// Store found result
__device__ void setPoolResultFound(int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5], unsigned int privateKey[8])
{
    CudaPoolDeviceResult r;

    r.block = blockIdx.x;
    r.thread = threadIdx.x;
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
        r.privateKey[i] = privateKey[i];
    }

    doPoolRMD160FinalRound(digest, r.digest);

    atomicListAdd(&r, sizeof(r));
}

/**
 * Perform scalar multiplication using double-and-add method with precomputed table
 *
 * This computes P = k * G where:
 * - k is the private key (256-bit scalar)
 * - G is the generator point
 * - The table contains G, 2G, 4G, ..., 2^255*G
 *
 * Algorithm:
 * For each bit i of k (from 0 to 255):
 *   if bit i is set:
 *     P = P + table[i]  (where table[i] = 2^i * G)
 */
__device__ void scalarMultiply(const unsigned int *privateKey, unsigned int *outX, unsigned int *outY)
{
    unsigned int *gxTable = _POOL_GX_TABLE[0];
    unsigned int *gyTable = _POOL_GY_TABLE[0];

    // Start with point at infinity
    unsigned int px[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    unsigned int py[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    bool isInf = true;

    // Process each bit of the private key
    for(int bitIdx = 0; bitIdx < 256; bitIdx++) {
        int wordIdx = 7 - (bitIdx / 32);  // Big-endian word order
        int bitPos = bitIdx % 32;

        unsigned int bit = (privateKey[wordIdx] >> bitPos) & 1;

        if(bit) {
            // Load table[bitIdx] = 2^bitIdx * G
            unsigned int gx[8], gy[8];
            for(int i = 0; i < 8; i++) {
                gx[i] = gxTable[bitIdx * 8 + i];
                gy[i] = gyTable[bitIdx * 8 + i];
            }

            if(isInf) {
                // First point, just copy
                for(int i = 0; i < 8; i++) {
                    px[i] = gx[i];
                    py[i] = gy[i];
                }
                isInf = false;
            } else {
                // Add points: P = P + table[bitIdx]
                unsigned int newX[8], newY[8];

                // Check if points are equal (need doubling instead of addition)
                bool equal = true;
                for(int i = 0; i < 8; i++) {
                    if(px[i] != gx[i]) {
                        equal = false;
                        break;
                    }
                }

                if(equal) {
                    // Point doubling
                    // lambda = (3*x^2 + a) / (2*y)  where a=0 for secp256k1
                    // newX = lambda^2 - 2*x
                    // newY = lambda*(x - newX) - y

                    unsigned int x2[8], twoY[8], lambda[8], lambda2[8];

                    // x^2
                    mulModP(px, px, x2);

                    // 3*x^2
                    unsigned int three[8] = {0,0,0,0,0,0,0,3};
                    mulModP(three, x2, x2);

                    // 2*y
                    addModP(py, py, twoY);

                    // lambda = 3*x^2 / 2*y
                    unsigned int twoYInv[8];
                    copyBigInt(twoY, twoYInv);
                    invModP(twoYInv);
                    mulModP(x2, twoYInv, lambda);

                    // lambda^2
                    mulModP(lambda, lambda, lambda2);

                    // newX = lambda^2 - 2*x
                    unsigned int twoX[8];
                    addModP(px, px, twoX);
                    subModP(lambda2, twoX, newX);

                    // newY = lambda*(x - newX) - y
                    unsigned int diff[8];
                    subModP(px, newX, diff);
                    mulModP(lambda, diff, newY);
                    subModP(newY, py, newY);
                } else {
                    // Point addition
                    // lambda = (y2 - y1) / (x2 - x1)
                    // newX = lambda^2 - x1 - x2
                    // newY = lambda*(x1 - newX) - y1

                    unsigned int dx[8], dy[8], lambda[8], lambda2[8];

                    // dx = gx - px
                    subModP(gx, px, dx);

                    // dy = gy - py
                    subModP(gy, py, dy);

                    // lambda = dy / dx
                    unsigned int dxInv[8];
                    copyBigInt(dx, dxInv);
                    invModP(dxInv);
                    mulModP(dy, dxInv, lambda);

                    // lambda^2
                    mulModP(lambda, lambda, lambda2);

                    // newX = lambda^2 - px - gx
                    subModP(lambda2, px, newX);
                    subModP(newX, gx, newX);

                    // newY = lambda*(px - newX) - py
                    unsigned int diff[8];
                    subModP(px, newX, diff);
                    mulModP(lambda, diff, newY);
                    subModP(newY, py, newY);
                }

                for(int i = 0; i < 8; i++) {
                    px[i] = newX[i];
                    py[i] = newY[i];
                }
            }
        }
    }

    for(int i = 0; i < 8; i++) {
        outX[i] = px[i];
        outY[i] = py[i];
    }
}

/**
 * Main kernel for key pool processing
 *
 * Each thread:
 * 1. Reads its assigned private key from the batch
 * 2. Computes the public key via scalar multiplication
 * 3. Hashes the public key
 * 4. Checks against target addresses
 */
__global__ void keyPoolKernel(int pointsPerThread, int compression)
{
    unsigned int *privatePtr = _POOL_PRIVATE_PTR[0];
    unsigned int *xPtr = _POOL_X_PTR[0];
    unsigned int *yPtr = _POOL_Y_PTR[0];

    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int privateKey[8];
        unsigned int pubX[8], pubY[8];
        unsigned int digest[5];

        // Read private key
        readPoolInt(privatePtr, i, privateKey);

        // Check if this slot has a valid key (not all zeros)
        bool hasKey = false;
        for(int j = 0; j < 8; j++) {
            if(privateKey[j] != 0) {
                hasKey = true;
                break;
            }
        }

        if(!hasKey) {
            continue;  // Empty slot, skip
        }

        // Perform scalar multiplication: pubKey = privateKey * G
        scalarMultiply(privateKey, pubX, pubY);

        // Store computed public key
        writePoolInt(xPtr, i, pubX);
        writePoolInt(yPtr, i, pubY);

        // Hash and check (uncompressed)
        if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
            hashPoolPublicKey(pubX, pubY, digest);

            if(checkHash(digest)) {
                setPoolResultFound(i, false, pubX, pubY, digest, privateKey);
            }
        }

        // Hash and check (compressed)
        if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {
            hashPoolPublicKeyCompressed(pubX, pubY[7] & 1, digest);

            if(checkHash(digest)) {
                setPoolResultFound(i, true, pubX, pubY, digest, privateKey);
            }
        }
    }
}

// Host function to launch kernel
void callKeyPoolKernel(int blocks, int threads, int pointsPerThread, int compression)
{
    keyPoolKernel<<<blocks, threads>>>(pointsPerThread, compression);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        throw cuda::CudaException(cudaGetErrorString(err));
    }
}
