#include <cstdlib> // rand
#include <cmath> // exp
#include "Neuron.hpp"

using namespace std;

/*--------------------------------------------------------------------*/
/*			Class definitions:	Neuron	/	Neurona                   */
/*--------------------------------------------------------------------*/

// Neuron Constructor
Neuron::Neuron( unsigned index, unsigned synapses, double eta, 
	double alpha )
{
	for (unsigned s = 0; s < synapses; ++s) 
	{
		neuron_synapse.push_back(Synapse());
		neuron_synapse.back().weight = rand()/double(RAND_MAX);
	}
	neuron_index = index;
	neuron_eta = eta;
	neuron_alpha = alpha;
}
		
// Method to set the signal of a neuron
void Neuron::setSignal( double value )
{
	neuron_signal = value;
}

// Method to get the signal of a neuron
double Neuron::getSignal( void )
{
	return neuron_signal;
}

// Method to get the weight of an incoming indexed synapse of a neuron
double Neuron::getWeight( unsigned index )
{ 
	return neuron_synapse[index].weight;
}

//Method to obtain the output value of a neuron during feedForward
void Neuron::feed( Layer &previousLayer )
{
	// Here we calculate the result for activation of this neuron
	// If using bias neurons, include them using size, not size - 1
	double sum = 0.0;
	for (unsigned n = 0; n < previousLayer.size() - 1; ++n)
	{
		//sumamos pesos * signal recibida y aplicamos func de activacion
		sum += previousLayer[n].getSignal() * 
			previousLayer[n].getWeight(n);
		neuron_signal = Neuron::activationFunction(sum);
	}
}

// Method to obtain the computed activation value of a neuron
double Neuron::activationFunction( double x )
{
	double sigmoid = 1 / (1 + exp(-x)); //for logical sigmoid
	return sigmoid;
}

// Method to obtain the derivative for activation function
double Neuron::derivativeFunction( double x )
{
	double denominator = (1 + exp(-x)) * (1 + exp(-x));
	double sigmoidDerivative = exp(-x) / denominator;
	return sigmoidDerivative;
}

// Method to calculate the output gradients
void Neuron::calculateOutputGradients( double target )
{
	double error = target - neuron_signal; // (t-y)
	double derivative = Neuron::derivativeFunction(neuron_signal);
	neuron_gradient = error * derivative; //(t-y) * s'(y)
}

// Method to calculate the hidden layer 
void Neuron::calculateHiddenGradients( Layer &nextLayer )
{
	double error = 0.0; // sumation of (t-y) * s'(y) connected to actual neuron
	for ( unsigned neuron = 0; neuron < nextLayer.size(); ++neuron )
	{
		// actual neuron contribution in nextLayer's error
		error += neuron_synapse[neuron].weight * 
			nextLayer[neuron].neuron_gradient;
	} 
	double derivative = Neuron::derivativeFunction(neuron_signal);
	neuron_gradient = error * derivative; // sum((t-y) * s'(y)) * s'(z)
}

void Neuron::updateInputWeights( Layer &previousLayer )
{
	// Update input weights for a neuron in actual layer
	for ( unsigned neuron = 0; neuron < previousLayer.size(); ++neuron )
	{
		Neuron &actualNeuron = previousLayer[neuron];
		//- eta * signal * gradient + alpha * previousWeightChange
		double gradientDescent = actualNeuron.neuron_eta *
			actualNeuron.neuron_signal * actualNeuron.neuron_gradient;
		double momentum = actualNeuron.neuron_alpha * 
			actualNeuron.neuron_synapse[neuron_index].deltaWeight;
		double newDeltaWeight = gradientDescent + momentum;
		
		actualNeuron.neuron_synapse[neuron_index].weight += newDeltaWeight;
		actualNeuron.neuron_synapse[neuron_index].deltaWeight = newDeltaWeight;
	}
}





