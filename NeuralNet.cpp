#include <cmath> // for actFunction and its derivative
#include "NeuralNet.hpp"

/*--------------------------------------------------------------------*/
/*			Class definitions:	NeuralNet	/	RedNeural             */
/*--------------------------------------------------------------------*/

// NeuralNet Constructor using hidden layer number and neuron number
NeuralNet::NeuralNet( unsigned hiddenLayers, unsigned hiddenNeurons,
	double eta, double alpha )
{
	
	vector<unsigned> structure;
	structure.push_back(8);// input neurons = 8
	for ( unsigned layer = 0; layer < hiddenLayers; ++layer )
	{
		structure.push_back(hiddenNeurons);
	}
	structure.push_back(5);// output neurons = 5
	unsigned netSize = structure.size();
	//create layers
	for ( unsigned layer = 0; layer < netSize; ++layer )
	{
		net_layers.push_back(Layer());
		unsigned synapses;
		if ( layer == structure.size() - 1 ) synapses = 0;
		else synapses = structure[layer + 1];
		// and add neurons for each layer
		for( unsigned neuron = 0; neuron < structure[layer]; ++neuron )
		{
			net_layers.back().push_back(Neuron(neuron,synapses,eta,alpha));
		}
	}
}

// Method that feeds the input values in the net to obtain op result
void NeuralNet::feedForward( vector<double> &inputVector )
{
	// inputVector: num1= 0 1 2 3 | num2 = 4 5 6 7
	
	// Assign input values from inputVector to inputLayer
	for ( unsigned input = 0; input < inputVector.size(); ++input )
	{
		net_layers[0][input].setSignal(inputVector[input]);
	}
	
	//Propagate the input throughout the neural net
	for ( unsigned layer = 1; layer < net_layers.size(); ++layer )
	{
		Layer &previousLayer = net_layers[layer - 1];
		for ( unsigned neuron = 0; neuron < net_layers[layer].size() - 1;
			++neuron)
		{
			net_layers[layer][neuron].feed(previousLayer);
		}
	}
}

// Method that propagates error and update net weights for learning
void NeuralNet::backPropagation( vector<double> &targetVector ) // 0 1 2 3 4
{
	// Calculate the neural net error processing outputs
	Layer &outputLayer = net_layers.back();
	net_error = 0.0;
	for ( unsigned neuron = 0; neuron < outputLayer.size(); ++neuron )
	{
		double diff = targetVector[neuron] - 
			outputLayer[neuron].getSignal();
		net_error += diff * diff;
	}
	net_error /= outputLayer.size();// MSE = sum( (t-y)^2 ) / N
	net_error = sqrt(net_error); // Root Mean Square Error
	
	// Calculate the neural network gradients
	for ( unsigned neuron = 0; neuron < outputLayer.size(); ++neuron )
	{
		outputLayer[neuron].calculateOutputGradients(targetVector[neuron]);
	}
	for ( unsigned layer = net_layers.size() - 2; layer > 0; --layer )
	{
		Layer &actualLayer = net_layers[layer];
		Layer &nextLayer = net_layers[layer + 1];
		for ( unsigned neuron = 0; neuron < actualLayer.size(); ++neuron )
		{
			actualLayer[neuron].calculateHiddenGradients(nextLayer);
		}
	}
	// Updates the neural network weights
	for ( unsigned layer = net_layers.size() - 1; layer > 0; --layer )
	{
		Layer &actualLayer = net_layers[layer];
		Layer &previousLayer = net_layers[layer -1];
		for ( unsigned neuron = 0; neuron < actualLayer.size(); ++neuron )
		{
			actualLayer[neuron].updateInputWeights(previousLayer);
		}
	}
}

// Method that obtains the result of a processed input vector
void NeuralNet::getResult( vector<double> &outputVector )
{
	outputVector.clear();
	Layer &outputLayer = net_layers[net_layers.size()-1];
	for ( unsigned neuron = 0; neuron < outputLayer.size(); ++neuron )
	{
		outputVector.push_back( outputLayer[neuron].getSignal() );
	}
}

double NeuralNet::getNetError( void )
{
	return net_error;
}


