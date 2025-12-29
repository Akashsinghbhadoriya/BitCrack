#ifndef _KEY_POOL_H
#define _KEY_POOL_H

#include <string>
#include <vector>
#include <random>
#include <mutex>
#include "secp256k1.h"

/**
 * KeyPool manages a pool of private keys generated from input file fragments.
 *
 * Key Construction Process:
 * 1. Read fragments from input file (one per line)
 * 2. Repeat each fragment N times (default: 3)
 * 3. Prepend configurable hex prefix
 * 4. Pad/format to valid 256-bit private key
 *
 * Supports random batch selection for GPU processing.
 */
class KeyPool {

public:
    struct Config {
        std::string inputFile;      // Path to input.txt with key fragments
        std::string prefix;         // Hex prefix to prepend (e.g., "8000000000000000")
        int chunkRepeat;            // Number of times to repeat each chunk (default: 3)
        bool randomMode;            // Enable random selection mode
        bool shuffle;               // Shuffle pool at startup

        Config() : chunkRepeat(3), randomMode(true), shuffle(true) {}
    };

private:
    Config _config;
    std::vector<secp256k1::uint256> _keyPool;       // All constructed keys
    std::vector<bool> _usedKeys;                     // Track which keys have been tested
    size_t _currentIndex;                            // Current position for sequential mode
    size_t _totalKeys;                               // Total number of keys in pool
    size_t _usedCount;                               // Number of keys already used

    std::mt19937_64 _rng;                           // Random number generator
    std::mutex _mutex;                               // Thread safety

    // Construct a private key from a fragment
    secp256k1::uint256 constructKey(const std::string &fragment);

    // Pad hex string to 64 characters (256 bits)
    std::string padToHex64(const std::string &hexStr);

    // Read fragments from file
    bool readFragments(const std::string &filename, std::vector<std::string> &fragments);

    // Shuffle the key pool
    void shufflePool();

public:
    KeyPool();
    KeyPool(const Config &config);
    ~KeyPool();

    // Initialize the pool from config
    bool init(const Config &config);
    bool init();

    // Get the next batch of keys for GPU processing
    // Returns false if no more keys available
    bool getNextBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut);

    // Get a specific number of random keys (may include duplicates if pool is small)
    bool getRandomBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut);

    // Get sequential batch (no randomization)
    bool getSequentialBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut);

    // Check if there are more keys to process
    bool hasMoreKeys() const;

    // Get total number of keys in pool
    size_t getTotalKeys() const;

    // Get number of remaining keys
    size_t getRemainingKeys() const;

    // Get number of used keys
    size_t getUsedKeys() const;

    // Reset the pool for reprocessing
    void reset();

    // Get a single key by index
    secp256k1::uint256 getKey(size_t index) const;

    // Mark a key as used
    void markUsed(size_t index);

    // Get all keys (for bulk GPU upload)
    const std::vector<secp256k1::uint256>& getAllKeys() const;

    // Validate that all keys are within valid range for secp256k1
    bool validateKeys();

    // Get config
    const Config& getConfig() const { return _config; }
};

#endif
