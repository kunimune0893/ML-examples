//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

constexpr int g_kMnistImageByteSize = 28 * 28;

// Helper struct for loading MNIST data
struct MnistImage
{
    unsigned int label;
    float image[g_kMnistImageByteSize];
};

// MNIST data files are big endian; we will need to swap them
void EndianSwap(unsigned int &x)
{
    x = (x >> 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x << 24);
}

// Load a single MNIST image from the raw data files - none of this is Arm NN-specific
std::unique_ptr<MnistImage[]> loadMnistImage(std::string dataDir, int image, int numImage = 1)
{
    std::vector<unsigned char> I(g_kMnistImageByteSize);
    unsigned int label = 0;

    std::string imagePath = dataDir + std::string("t10k-images-idx3-ubyte");
    std::string labelPath = dataDir + std::string("t10k-labels-idx1-ubyte");

    std::ifstream imageStream(imagePath, std::ios::binary);
    std::ifstream labelStream(labelPath, std::ios::binary);

    if (!imageStream.is_open())
    {
        std::cerr << "Failed to load " << imagePath << std::endl;
        return nullptr;
    }
    if (!labelStream.is_open())
    {
        std::cerr << "Failed to load " << labelPath << std::endl;
        return nullptr;
    }

    unsigned int magic, num, row, col;

    // check the files have the correct header
    imageStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x03080000)
    {
        std::cerr << "Failed to read " << imagePath << std::endl;
        return nullptr;
    }
    labelStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x01080000)
    {
        std::cerr << "Failed to read " << labelPath << std::endl;
        return nullptr;
    }

    // Endian swap image and label file - All the integers in the files are stored in MSB first(high endian) format,
    // hence need to flip the bytes of the header if using it on Intel processors or low-endian machines
    labelStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    imageStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    EndianSwap(num);
    imageStream.read(reinterpret_cast<char*>(&row), sizeof(row));
    EndianSwap(row);
    imageStream.read(reinterpret_cast<char*>(&col), sizeof(col));
    EndianSwap(col);

    std::unique_ptr<MnistImage[]> ret(new MnistImage[10000]);
    
    imageStream.seekg(image * g_kMnistImageByteSize, std::ios_base::cur);
    labelStream.seekg(image, std::ios_base::cur);
    
    for (unsigned int ii = 0; ii < numImage; ++ii)
    {
	    // read image and label into memory
	    imageStream.read(reinterpret_cast<char*>(&I[0]), g_kMnistImageByteSize);
	    labelStream.read(reinterpret_cast<char*>(&label), 1);
		
	    if (!imageStream.good())
	    {
	        std::cerr << "Failed to read " << imagePath << std::endl;
	        return nullptr;
	    }
	    if (!labelStream.good())
	    {
	        std::cerr << "Failed to read " << labelPath << std::endl;
	        return nullptr;
	    }
	
	    // store image and label in MnistImage
	    ret[ii].label = label;
	
	    for (unsigned int i = 0; i < col * row; ++i)
	    {
	        ret[ii].image[i] = static_cast<float>(I[i]) / 255;
#if 0
	        if( ii == 0 ){
	            printf( "%f ", ret[ii].image[i] );
	        }
#endif
	    }
	}
	
    return ret;
}
