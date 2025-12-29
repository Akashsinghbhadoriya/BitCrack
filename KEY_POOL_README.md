# BitCrack Key Pool Mode

This modification adds a custom key generation mode to BitCrack that allows you to:
1. Read key fragments from an input file
2. Repeat each fragment a configurable number of times
3. Add a configurable hex prefix
4. Select keys randomly (not sequentially)

## New Command Line Options

```
--keypool FILE          Read key fragments from FILE (enables key pool mode)
--prefix HEX            Hex prefix to prepend to each key (e.g., 8000000000000000)
--chunk-repeat N        Repeat each fragment N times (default: 3)
--sequential            Use sequential selection instead of random
```

## How It Works

### Key Construction Process

1. **Read Fragments**: Each line in the input file is treated as a fragment
2. **Repeat**: The fragment is repeated N times (configurable with `--chunk-repeat`)
3. **Add Prefix**: The hex prefix is prepended to the repeated fragment
4. **Pad to 256 bits**: The result is padded with leading zeros to create a valid 256-bit private key

### Example

Input file (`input.txt`):
```
abc123
def456
```

With options `--prefix 8000 --chunk-repeat 3`:

Fragment `abc123` becomes:
1. Repeat 3x: `abc123abc123abc123`
2. Add prefix: `8000abc123abc123abc123`
3. Pad to 64 hex chars: `0000000000000000000000000000000000000000008000abc123abc123abc123`

### Input File Format

- One fragment per line
- Lines starting with `#` are treated as comments
- Empty lines are ignored
- Fragments can be hex strings or ASCII text
  - Hex strings are used directly
  - ASCII text is converted to hex (each character → 2 hex digits)

## Usage Examples

### Basic Usage

```bash
./cuBitCrack --keypool input.txt -i targets.txt -o found.txt
```

### With Custom Prefix and Repeat Count

```bash
./cuBitCrack --keypool input.txt --prefix 8000000000000000 --chunk-repeat 5 -i targets.txt
```

### Sequential Mode (No Randomization)

```bash
./cuBitCrack --keypool input.txt --sequential -i targets.txt
```

### Full Example

```bash
./cuBitCrack \
    --keypool fragments.txt \
    --prefix 00000000000000000000000000000001 \
    --chunk-repeat 3 \
    -i targets.txt \
    -o results.txt \
    -c \
    -d 0
```

## Performance Considerations

Key pool mode is slower than standard sequential mode because:

1. **Full Scalar Multiplication**: Each key requires a complete k*G computation (~256 point operations) instead of a single point addition
2. **CPU-GPU Transfer**: Keys must be transferred from CPU to GPU each batch
3. **No Sequential Optimization**: Cannot use the efficient point addition trick

**Expected Performance**: 20-50% of original speed

| GPU | Original Speed | Key Pool Speed (Expected) |
|-----|---------------|---------------------------|
| GTX 1070 | 180-245 MKey/s | 50-100 MKey/s |
| GTX 1080 | ~343 MKey/s | 100-150 MKey/s |
| RTX 3090 | 800+ MKey/s | 200-400 MKey/s |

## Building

The key pool feature is included in the standard build:

```bash
make BUILD_CUDA=1
```

## Technical Details

### New Files Added

- `KeyFinderLib/KeyPool.h` - Key pool class declaration
- `KeyFinderLib/KeyPool.cpp` - Key pool implementation
- `CudaKeySearchDevice/CudaKeyPoolDevice.h` - GPU device for pool mode
- `CudaKeySearchDevice/CudaKeyPoolDevice.cpp` - CPU-side implementation
- `CudaKeySearchDevice/CudaKeyPoolDevice.cu` - CUDA kernel for pool mode

### Modified Files

- `KeyFinder/main.cpp` - Added new CLI options and pool mode runner

### Algorithm

The GPU kernel uses double-and-add scalar multiplication with a precomputed table:

1. Precompute table: G, 2G, 4G, 8G, ..., 2^255*G
2. For each private key k:
   - Start with point at infinity
   - For each bit i of k (0 to 255):
     - If bit i is set: P = P + table[i]
   - Result P = k * G

This allows computing any k*G in exactly 256 point additions (on average, 128 additions for random keys).
