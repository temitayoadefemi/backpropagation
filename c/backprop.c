#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helper.h"
#include "activation.h"
#include <math.h>


#define INPUT_LAYER 2
#define HIDDEN_LAYER 10
#define OUTPUT_LAYER 1

#define LEARNING_RATE 0.5
#define EPOCHS 10000


double w1[INPUT_LAYER][HIDDEN_LAYER];
double b1[HIDDEN_LAYER];
double w2[HIDDEN_LAYER][OUTPUT_LAYER];
double b2[OUTPUT_LAYER];
double hidden_input[HIDDEN_LAYER];
double hidden_output[HIDDEN_LAYER];
double final_input[OUTPUT_LAYER];
double final_output[OUTPUT_LAYER];
double output_delta[OUTPUT_LAYER];
double hidden_error[HIDDEN_LAYER];
double hidden_delta[HIDDEN_LAYER];

void init_weights() {
    srand(time(NULL));

    for (int i = 0; i < INPUT_LAYER; i++)
        for (int j = 0; j < HIDDEN_LAYER; j++)
            w1[i][j] = box_muller_transform(0, 0.1);

    for (int i = 0; i < HIDDEN_LAYER; i++)
        for (int j = 0; j < OUTPUT_LAYER; j++)
            w2[i][j] = box_muller_transform(0, 0.1);

    for (int i = 0; i < HIDDEN_LAYER; i++) 
        b1[i] = 0;

    for (int i = 0; i < OUTPUT_LAYER; i++) 
        b2[i] = 0;
}


void forward_pass(double *x) {
    for (int i = 0; i < HIDDEN_LAYER; i++) {
        hidden_input[i] = 0;
        for (int j = 0; j < INPUT_LAYER; j++) {
            hidden_input[i] += w1[j][i] * x[j];
        }
        hidden_input[i] += b1[i];
        hidden_output[i] = sigmoid(hidden_input[i]);
    }

    for (int i = 0; i < OUTPUT_LAYER; i++) {
        final_input[i] = 0;
        for (int j = 0; j < HIDDEN_LAYER; j++) {
            final_input[i] += w2[j][i] * hidden_output[j];
        }
        final_input[i] += b2[i];
        final_output[i] = sigmoid(final_output[i]);
    }
}


void backward_pass(double *x, double target) {
    double output_error = target - final_output[0];
    double output_delta = output_error * sigmoid_derivative(final_output[0]);

    for (int i = 0; i < HIDDEN_LAYER; i++) {
        double hidden_error = output_delta * w2[i][0];
        double hidden_delta = hidden_error * sigmoid_derivative(hidden_output[i]);

        // Update hidden to output weights and biases
        w2[i][0] += LEARNING_RATE * hidden_output[i] * output_delta;
        b2[0] += LEARNING_RATE * output_delta;

        // Update input to hidden weights and biases
        for (int j = 0; j < INPUT_LAYER; j++) {
            w1[j][i] += LEARNING_RATE * x[j] * hidden_delta;
        }
        b1[i] += LEARNING_RATE * hidden_delta;
    }
}


int main() {
    init_weights();
    double X[4][2] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    double y[4] = {0, 1, 1, 0};

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward_pass(X[i]);
            backward_pass(X[i], y[i]);
        }
    }

        // Test the network after training
    for (int i = 0; i < 4; i++) {
        forward_pass(X[i]);
        printf("Input: [%d, %d] - Predicted: %f\n", (int)X[i][0], (int)X[i][1], final_output[0]);
    }

    return 0;

}