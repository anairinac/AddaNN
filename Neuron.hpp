#if ! defined _NEURON_H
#define _NEURON_H 1

#include <vector> // vector

using namespace std;

/*--------------------------------------------------------------------*/

//	Synapse:	connection structure between neurons
//				estructura de conexion entre neuronas
//				weight = peso de conexion
//				bpWeight = peso de conexion en el backProp
struct Synapse
{
	double weight;
	double deltaWeight;
};

/*--------------------------------------------------------------------*/

class Neuron;

/*--------------------------------------------------------------------*/

//	Layer:		neuron vector as network layer
//				vector de neuronas como capa de la red
//				input layer = capa de entrada
//				hidden layer(s) = capa(s) oculta(s)
//				output layer = capa de salida
typedef vector<Neuron> Layer;

/*--------------------------------------------------------------------*/
/*				Class declaration:	Neuron	/	Neurona               */
/*--------------------------------------------------------------------*/

class Neuron
{
	public:
		//-- Constructor with index within layer
		Neuron( unsigned index, unsigned synapses, double eta, 
			double alpha );
		
		//-- Public methods
		void setSignal( double value );
		double getSignal( void );
		double getWeight( unsigned index );
		void feed( Layer &previousLayer );
		
		//-- Methods for back propagation in training
		void calculateOutputGradients( double target );
		void calculateHiddenGradients( Layer &nextLayer );
		void updateInputWeights( Layer &previousLayer );
		
		
	private:
		//-- Attributes
		unsigned neuron_index; // index within layer
		double neuron_signal; // signal to send to other neurons
		vector<Synapse> neuron_synapse; // synapses with weights
		
		//gradient descent attributes for backprop
		double neuron_gradient; // neuron weight gradient
		double neuron_eta; // learning rate ]0.0, 0.9[
		double neuron_alpha; // momentum constant
		
		//-- Private methods
		static double activationFunction( double x ); // ACTIVATION FUNC
		static double derivativeFunction( double value );
};

#endif
