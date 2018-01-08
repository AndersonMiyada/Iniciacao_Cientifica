
package visaocomputacional;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import visaocomputacional.VetorCaracteristica;

public class Principal {
/*        
	public static double MatInput[][] = { {0, 0, 0, 1, 0},
                                              {0, 1, 1, 1, 0},
                                              {0, 1, 0, 1, 0},
                                              {0, 1, 1, 1, 0},
                                              {0, 1, 0, 1, 0}
                                            };

	
	public static double MatIdeal[][] = { {0, 0, 0, 0, 0}, 
                                              {0, 1, 1, 1, 0}, 
                                              {0, 1, 0, 1, 0},
                                              {0, 1, 1, 1, 0},
                                              {0, 1, 0, 1, 0}
                                            };
*/
	public static double MatInput[][] = {   {0,1,1,1,0,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0},// letra A
                                                {0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,1,0,0},// Letra B
                                                {0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0},// Letra C
                                                {0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,0},// Letra D
                                                {0,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0},// Letra E
                                                {0,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0},// Letra F
                                                {0,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,0,0},// Letra G
                                                {0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0,0},// Letra H
                                                {0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},// Letra I
                                                {0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0},// Letra J
                                                {0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1},// Letra K
                                                {0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0},// Letra L
                                                {1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1},// Letra M
                                                {1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1},// Letra N
                                                {0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0},// Letra O
                                                {0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0},// Letra P
                                                {0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1},// Letra Q
                                                {0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0},// Letra R
                                                {0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0},// Letra S
                                                {0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},// Letra T
                                                {0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0},// Letra U
                                                {1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0},// Letra V
                                                {1,0,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1},// Letra W
                                                {1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1},// Letra X
                                                {1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},// Letra Y
                                                {1,1,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1} // Letra Z
                                               };

	
	//public static double MatIdeal[][] = { {0.0}, {1.0}  };
	
	public static void main(String[] args) {
		
		// cria uma instância de um neurônio básico;
		BasicNetwork network = new BasicNetwork();
                //Adiciona uma camada básica de entrada com os parâmetros
                //(Função de Ativação, possui viés, quantidade de neurônios na camada) 
		network.addLayer(new BasicLayer(null,true,25));
                //Adiciona uma camada intermediária criando neurônios internos com tipo de ativação e quantidade
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,25));
                //cria uma camada de saída com um neuronio
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));
                //retorna a estrutura criada e a finaliza
		network.getStructure().finalizeStructure();
                // RESETA A ESTRUTURA
		network.reset();
                
                
            

		// MLDataSet: Uma classe abstrata que armazena os resultados de treinamento;
                // BasicMLDataSet: Armazena dados em um ArrayList;
		MLDataSet trainingSet = new BasicMLDataSet(MatInput, VetorCaracteristica.matIdeal);
		
		// Contrói o treinamento da rede
		final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
                // Variável para armazenar a quantidade de treinamentos
		int epoch = 1;
                // Loop de treinamento
		do {
                        //Realiza uma iteração de treinamento
			train.iteration();
                        // Mostra a época atual de treinamento e o valor do erro de treinamento
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
                        // Incrementa a época
			epoch++;
                // Condição de parada do treinamento
		} while(train.getError() > 0.001);
                
                // Mostra o resultado do treinamento
		System.out.println("Neural Network Results:");
                
		for(MLDataPair pair: trainingSet ) {
			final MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
					+ ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
		}
                
                // Finaliza o treinamento
		train.finishTraining();

		
                
                // Vetor de características para reconhecimento;
		BasicMLData in = new BasicMLData(new double[] {0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0});// LetraC
                
                // Recebe o vetor de característica e analisa se é reconhecido ou não; 
                MLData output = network.compute(in);
                // Mostra o resultado 
                if(output.getData(0) < 0.1)
                    System.out.println("Reconhecido: " + output.getData(0));
                else
                    System.out.println("Não reconhecido!");
               
                //Encerra a instância do Encog;
		Encog.getInstance().shutdown();
	}
    };