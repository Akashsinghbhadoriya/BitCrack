#include "KeyPool.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "Logger.h"

KeyPool::KeyPool()
    : _currentIndex(0), _totalKeys(0), _usedCount(0)
{
    // Seed the random number generator with current time
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    _rng.seed(seed);
}

KeyPool::KeyPool(const Config &config)
    : _config(config), _currentIndex(0), _totalKeys(0), _usedCount(0)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    _rng.seed(seed);
}

KeyPool::~KeyPool()
{
}

bool KeyPool::readFragments(const std::string &filename, std::vector<std::string> &fragments)
{
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        Logger::log(LogLevel::Error, "Unable to open input file: " + filename);
        return false;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");

        if (start != std::string::npos && end != std::string::npos) {
            std::string fragment = line.substr(start, end - start + 1);

            // Skip empty lines and comment lines (starting with #)
            if (!fragment.empty() && fragment[0] != '#') {
                fragments.push_back(fragment);
            }
        }
    }

    inFile.close();
    return true;
}

std::string KeyPool::padToHex64(const std::string &hexStr)
{
    std::string result = hexStr;

    // Remove 0x prefix if present
    if (result.length() >= 2 && result[0] == '0' && (result[1] == 'x' || result[1] == 'X')) {
        result = result.substr(2);
    }

    // Truncate if too long, pad if too short
    if (result.length() > 64) {
        result = result.substr(result.length() - 64);
    } else if (result.length() < 64) {
        result = std::string(64 - result.length(), '0') + result;
    }

    return result;
}

secp256k1::uint256 KeyPool::constructKey(const std::string &fragment)
{
    // Step 1: Repeat the fragment N times
    std::string repeated;
    for (int i = 0; i < _config.chunkRepeat; i++) {
        repeated += fragment;
    }

    // Step 2: Convert to hex if not already hex
    std::string hexStr;
    bool isHex = true;
    for (char c : repeated) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            isHex = false;
            break;
        }
    }

    if (isHex) {
        hexStr = repeated;
    } else {
        // Convert ASCII to hex
        std::stringstream ss;
        for (unsigned char c : repeated) {
            ss << std::hex << std::setfill('0') << std::setw(2) << (int)c;
        }
        hexStr = ss.str();
    }

    // Step 3: Prepend the prefix
    std::string prefix = _config.prefix;
    if (prefix.length() >= 2 && prefix[0] == '0' && (prefix[1] == 'x' || prefix[1] == 'X')) {
        prefix = prefix.substr(2);
    }

    std::string combined = prefix + hexStr;

    // Step 4: Pad to 64 hex characters (256 bits)
    std::string padded = padToHex64(combined);

    // Step 5: Create uint256 from hex string
    try {
        return secp256k1::uint256(padded);
    } catch (...) {
        Logger::log(LogLevel::Error, "Invalid key construction from fragment: " + fragment);
        return secp256k1::uint256(1); // Return 1 as fallback
    }
}

void KeyPool::shufflePool()
{
    std::shuffle(_keyPool.begin(), _keyPool.end(), _rng);
}

bool KeyPool::init(const Config &config)
{
    _config = config;
    return init();
}

bool KeyPool::init()
{
    std::lock_guard<std::mutex> lock(_mutex);

    _keyPool.clear();
    _usedKeys.clear();
    _currentIndex = 0;
    _usedCount = 0;

    // Read fragments from file
    std::vector<std::string> fragments;
    if (!readFragments(_config.inputFile, fragments)) {
        return false;
    }

    if (fragments.empty()) {
        Logger::log(LogLevel::Error, "No fragments found in input file");
        return false;
    }

    Logger::log(LogLevel::Info, "Loaded " + std::to_string(fragments.size()) + " fragments from " + _config.inputFile);
    Logger::log(LogLevel::Info, "Chunk repeat: " + std::to_string(_config.chunkRepeat) + "x");
    Logger::log(LogLevel::Info, "Prefix: " + (_config.prefix.empty() ? "(none)" : _config.prefix));

    // Construct keys from fragments
    for (const auto &fragment : fragments) {
        secp256k1::uint256 key = constructKey(fragment);

        // Validate key is within range (1 to N-1)
        if (!key.isZero() && key.cmp(secp256k1::N) < 0) {
            _keyPool.push_back(key);
        } else {
            Logger::log(LogLevel::Info, "Skipping invalid key from fragment: " + fragment);
        }
    }

    if (_keyPool.empty()) {
        Logger::log(LogLevel::Error, "No valid keys generated from fragments");
        return false;
    }

    _totalKeys = _keyPool.size();
    _usedKeys.resize(_totalKeys, false);

    Logger::log(LogLevel::Info, "Generated " + std::to_string(_totalKeys) + " valid keys");

    // Shuffle if enabled
    if (_config.shuffle) {
        shufflePool();
        Logger::log(LogLevel::Info, "Key pool shuffled");
    }

    return true;
}

bool KeyPool::getNextBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut)
{
    if (_config.randomMode) {
        return getRandomBatch(batchSize, keysOut);
    } else {
        return getSequentialBatch(batchSize, keysOut);
    }
}

bool KeyPool::getRandomBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut)
{
    std::lock_guard<std::mutex> lock(_mutex);

    keysOut.clear();

    if (_keyPool.empty()) {
        return false;
    }

    // Find unused keys
    std::vector<size_t> unusedIndices;
    for (size_t i = 0; i < _totalKeys; i++) {
        if (!_usedKeys[i]) {
            unusedIndices.push_back(i);
        }
    }

    if (unusedIndices.empty()) {
        return false;
    }

    // Shuffle unused indices
    std::shuffle(unusedIndices.begin(), unusedIndices.end(), _rng);

    // Take up to batchSize keys
    size_t count = std::min(batchSize, unusedIndices.size());
    for (size_t i = 0; i < count; i++) {
        size_t idx = unusedIndices[i];
        keysOut.push_back(_keyPool[idx]);
        _usedKeys[idx] = true;
        _usedCount++;
    }

    return true;
}

bool KeyPool::getSequentialBatch(size_t batchSize, std::vector<secp256k1::uint256> &keysOut)
{
    std::lock_guard<std::mutex> lock(_mutex);

    keysOut.clear();

    if (_currentIndex >= _totalKeys) {
        return false;
    }

    size_t endIndex = std::min(_currentIndex + batchSize, _totalKeys);

    for (size_t i = _currentIndex; i < endIndex; i++) {
        keysOut.push_back(_keyPool[i]);
        _usedKeys[i] = true;
        _usedCount++;
    }

    _currentIndex = endIndex;
    return true;
}

bool KeyPool::hasMoreKeys() const
{
    return _usedCount < _totalKeys;
}

size_t KeyPool::getTotalKeys() const
{
    return _totalKeys;
}

size_t KeyPool::getRemainingKeys() const
{
    return _totalKeys - _usedCount;
}

size_t KeyPool::getUsedKeys() const
{
    return _usedCount;
}

void KeyPool::reset()
{
    std::lock_guard<std::mutex> lock(_mutex);

    std::fill(_usedKeys.begin(), _usedKeys.end(), false);
    _currentIndex = 0;
    _usedCount = 0;

    if (_config.shuffle) {
        shufflePool();
    }
}

secp256k1::uint256 KeyPool::getKey(size_t index) const
{
    if (index < _totalKeys) {
        return _keyPool[index];
    }
    return secp256k1::uint256(1);
}

void KeyPool::markUsed(size_t index)
{
    std::lock_guard<std::mutex> lock(_mutex);

    if (index < _totalKeys && !_usedKeys[index]) {
        _usedKeys[index] = true;
        _usedCount++;
    }
}

const std::vector<secp256k1::uint256>& KeyPool::getAllKeys() const
{
    return _keyPool;
}

bool KeyPool::validateKeys()
{
    for (const auto &key : _keyPool) {
        if (key.isZero() || key.cmp(secp256k1::N) >= 0) {
            return false;
        }
    }
    return true;
}
