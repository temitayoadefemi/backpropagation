#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define INPUT_LAYER 2
#define HIDDEN_LAYER 10
#define OUTPUT_LAYER 1

void box_muller_transform(double mean, double stddev) {
    static double n2 = 0.0;
    static int n2_cached = 0;

    if (!n2_cached) {
        double x, y, r;
    do{   
        x = 2.0*rand()/RAND_MAX - 1;
        y = 2.0*rand()/RAND_MAX - 1;
        r = x*x + y*y;
    } while (r == 0 || r > 1.0);
        double d = sqrt(-2.0*log(r)/r);
        double n1 = x*d;
        n2 = y*d;
        double result = n1*stddev + mean;
        n2_cached = 1;
        return result;
    } else {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}



