//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfParser/ITfParser.hpp"

#include "mnist_loader.hpp"


// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}

#include <boost/log/trivial.hpp>
#include "armnn/Utils.hpp"
int main(int argc, char** argv)
{
	int testBatch = 1, testCorrect = 0;
	
	// ロガー初期化
	armnn::LogSeverity level = armnn::LogSeverity::Debug;
	armnn::ConfigureLogging( true, true, level );
	
	// 引数表示
	for( int ii = 0; ii < argc; ii++ ){
		BOOST_LOG_TRIVIAL(fatal) << "argv[" << ii << "]: " << argv[ii];
	}
	
	if( argc > 1 ){
		testBatch = atoi( argv[1] );
		if( testBatch == 0 ){
			// 読み取りエラー
			BOOST_LOG_TRIVIAL(warning) << "can't parse: testBatch = " << testBatch;
		}
	}
	
    // Import the TensorFlow model. Note: use CreateNetworkFromBinaryFile for .pb files.
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
#if 0
    armnn::INetworkPtr network = parser->CreateNetworkFromTextFile("model/simple_mnist_tf.prototxt",
                                                                   { {"Placeholder", {1, 784, 1, 1}} },
                                                                   { "Softmax" });

    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("Placeholder");
    armnnTfParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("Softmax");
#else
    armnn::INetworkPtr network = parser->CreateNetworkFromTextFile("model/cnn-model-prob.pbtxt",
                                                                   { {"tf_x", {1, 784, 1, 1}} },
                                                                   { "probabilities" });
    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("tf_x");
    armnnTfParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("probabilities");
#endif

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
	armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuAcc);
//	armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuRef);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, runtime->GetDeviceSpec());

    // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    // Load a test image and its correct label
    std::string dataDir = "data/";
    int testImageIndex = 0;
    
    std::unique_ptr<MnistImage[]> input = loadMnistImage(dataDir, testImageIndex, 10000);
    if (input == nullptr)
        return 1;
    
    for( ; testImageIndex < testBatch; testImageIndex++ )
    {
    	// Run a single inference on the test image
    	std::array<float, 10> output;
    	armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
    	                                             MakeInputTensors(inputBindingInfo, &input[testImageIndex].image[0]),
    	                                             MakeOutputTensors(outputBindingInfo, &output[0]));
		
    	// Convert 1-hot output to an integer label and print
    	int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    	
    	if( label == input[testImageIndex].label ){ testCorrect++; }
    	//std::cout << label << " ";
    }
	std::cout << "Accuracy: " << testCorrect << " (" << testBatch << ")" << std::endl;
    
//	std::cout << "Predicted: " << label << std::endl;
//	std::cout << "   Actual: " << input->label << std::endl;
    
    return 0;
}
