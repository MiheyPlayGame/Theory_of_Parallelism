#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>

#define _USE_MATH_DEFINES
#define MOD 10000000

#ifdef USE_FLOAT
    typedef float real_t;
#else
    typedef double real_t;
#endif

int main(){
    std::vector<real_t> sinus(MOD);
    real_t s = 0.0;

    for (int i = 0; i < MOD; i++){
        sinus[i] = sin(2*M_PI*i/MOD);
        s += sinus[i];
    }

    printf("sum = %f\n", s);

    return 0;
}