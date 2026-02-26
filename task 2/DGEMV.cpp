#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

int m = 100;
int n = m;

void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        perror("xmalloc");
        exit(EXIT_FAILURE); // Завершение при ошибке
    }
    return ptr;
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n){
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
        c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(){
    double *a, *b, *c;
    a = (double*)xmalloc(sizeof(*a) * m * n);
    b = (double*)xmalloc(sizeof(*b) * n);
    c = (double*)xmalloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        a[i * n + j] = i + j;
    }
    for (int j = 0; j < n; j++)
    b[j] = j;
    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_parallel(){
    double *a, *b, *c;
    // Allocate memory for 2-d array a[m, n]
    a = (double*)xmalloc(sizeof(*a) * m * n);
    b = (double*)xmalloc(sizeof(*b) * n);
    c = (double*)xmalloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        a[i * n + j] = i + j;
    }
    for (int j = 0; j < n; j++)
    b[j] = j;
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv){
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %d MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    run_serial();
    run_parallel();

return 0;
}