
#include "network_evaluate.h"



float linear(float num) {
	return num;
}



float sigmoid(float num) {
	return 1 / (1 + exp(-num));
}



float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static int state_dim = 22;
static int action_dim = 4;
static int goal_dim = 18;
static int hidden_dim = 16;
static const float linear_1_weight[16][40] = {0,};
static const float linear_1_bias[16] = {0};
static const float linear_rnn_weight[16][26] = {0,};
static const float linear_rnn_bias[16] = {0};
static const float rnn_weight_ih[16][16] = {0,};
static const float rnn_bias_ih[16] = {0};
static const float rnn_weight_hh[16][16] = {0,};
static const float rnn_bias_hh[16] = {0};
static const float linear_3_weight[16][64] = {0,};
static const float linear_3_bias[16] = {0};
static const float linear_4_weight[16][16] = {0,};
static const float linear_4_bias[16] = {0};
static const float linear_mean_weight[4][16] = {0,};
static const float linear_mean_bias[4] = {0};

static float linear_input[40];
static float rnn_input[26];
static float output_linear1[16];
static float output_linear_rnn[16];
static float output_hidden[16] = {0.0};
static float dummy[16] = {0.0};
static float output_cat[64];
static float output_linear3[16];
static float output_linear4[16];
static float output_action[4]={0,0,0,0};
static const float goal[18]={0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0};


	void networkEvaluate(struct control_t_n *control_n, const float *state_array) {
	
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
			rnn_input[i] = state_array[i];
		}
		for (int i = 0; i < goal_dim; i++) {
			linear_input[state_dim + i] = goal[i];
		}
		for (int i = 0; i < action_dim; i++) {
			rnn_input[state_dim + i] = output_action[i];
		}

	
		for (int i = 0; i < 16; i++) {
			output_linear1[i] = 0;
			for (int j = 0; j < 40; j++) {
				output_linear1[i] += linear_input[j] * linear_1_weight[i][j];
			}
			output_linear1[i] += linear_1_bias[i];
			output_linear1[i] = tanhf(output_linear1[i]);
		}
	

		for (int i = 0; i < 16; i++) {
			output_linear_rnn[i] = 0;
			for (int j = 0; j < 26; j++) {
				output_linear_rnn[i] += rnn_input[j] * linear_rnn_weight[i][j];
			}
			output_linear_rnn[i] += linear_rnn_bias[i];
			output_linear_rnn[i] = tanhf(output_linear_rnn[i]);
		}
	

		for (int i = 0; i < 16; i++) {
			dummy[i] = 0;
			for (int j = 0; j < 16; j++) {
				dummy[i] += output_linear_rnn[j] * rnn_weight_ih[i][j] + output_hidden[j] * rnn_weight_hh[i][j];
			}
			dummy[i] = dummy[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < 16; i++) {
			output_hidden[i] = tanhf(dummy[i]);
		}
	

		for (int i = 0; i < hidden_dim; i++) {
			output_cat[i] = output_linear1[i];
			output_cat[hidden_dim + i] = output_hidden[i];
		}
	

		for (int i = 0; i < 16; i++) {
			output_linear3[i] = 0;
			for (int j = 0; j < 64; j++) {
				output_linear3[i] += output_cat[j] * linear_3_weight[i][j];
			}
			output_linear3[i] += linear_3_bias[i];
			output_linear3[i] = tanhf(output_linear3[i]);
		}
	

		for (int i = 0; i < 16; i++) {
			output_linear4[i] = 0;
			for (int j = 0; j < 16; j++) {
				output_linear4[i] += output_linear3[j] * linear_4_weight[i][j];
			}
			output_linear4[i] += linear_4_bias[i];
			output_linear4[i] = tanhf(output_linear4[i]);
		}
	

		for (int i = 0; i < 4; i++) {
			output_action[i] = 0;
			for (int j = 0; j < 16; j++) {
				output_action[i] += output_linear4[j] * linear_mean_weight[i][j];
			}
			output_action[i] += linear_mean_bias[i];
			output_action[i] = tanhf(output_action[i]);
		}
	

		control_n->thrust_0 = output_action[0];
		control_n->thrust_1 = output_action[1];
		control_n->thrust_2 = output_action[2];
		control_n->thrust_3 = output_action[3];	
	
	}
	