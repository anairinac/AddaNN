#if ! defined _NEURALNET_H
#define _NEURALNET_H 1

#include <vector>
#include "Neuron.hpp"

/*--------------------------------------------------------------------*/
/*			Class declaration:	NeuralNet	/	RedNeural             */
/*--------------------------------------------------------------------*/

class NeuralNet
{
	public:
		//-- Constructor
		NeuralNet( unsigned hiddenLayers, unsigned neurons, double eta,
			double alpha );
		void feedForward( vector<double> &inputVector );// 0 1 2 3 | 4 5 6 7
		void backPropagation( vector<double> &targetVector ); // 0 1 2 3 4
		void getResult( vector<double> &outputVector );
		double getNetError( void );
		
	private:
		vector<Layer> net_layers;
		double net_error; 
};

#endif
