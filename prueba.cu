
#include <omp.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>

// ./prog 0 10 13 1 10 32 1

// Crear linea
void row_line(int n) {
        printf("\n");
        for (int i=0; i<n; i++){
                printf("---");
        }
        printf("\n");
}

// Imprimir Matriz
void printmat(int n, int *m){

        for(int i=0; i<n; ++i){
                for(int j=0; j<n; ++j){
                        printf(" %d ", m[i*n+j]);
                }
                row_line(n);
        }
}

//Igualar matrices
void igualar(int n, int *a, int *b){
        for (int i=0; i<n; i++){
                for (int j=0; j<n; j++){
                        a[i*n+j]=b[i*n+j];
                }
        }
}

//Crear Matriz
void crearMatriz(int n, int *m, int seed){
        #pragma omp parallel shared(m)
        {
                int tid = omp_get_thread_num();
                int nt = omp_get_num_threads();
                long elems = (long)n * n;
                long chunk = elems/nt;
                long start = tid*chunk;
                long end = start + chunk;
                std::mt19937 mt(seed+tid);
                std::uniform_int_distribution<>  dist(0, 1);
                for(int k=start; k<n && k<end; ++k){
                        for (int j=start; j<n && j<end; ++j){
                                m[k*n+j] = dist(mt);
                        }
                }
        }
}

// Funciones para cpu sim

int count_live_neighbour_cell(int n, int *a, int r, int c){
        int count = 0;
        for (int i = r-1; i <= r+1; i++) {
                for (int j = c-1; j <= c+1; j++) {
                        if ((i==r && j==c) || (i<0 || j<0) || (i>=n || j>=n)) { continue; }
                        if (a[i*n+j] == 1) { count++; }
                }
        }
        return count;
}

void cpu_sim(int n, int *A, int *B){
        #pragma omp parallel for
        for (int i=0; i<n; i++) {
                for (int j=0; j<n; j++) {
                        int neighbour_live_cell = count_live_neighbour_cell(n, A, i, j);
                        if (A[i*n+j]==1 && (neighbour_live_cell==2 || neighbour_live_cell==3)) {
                                B[i*n+j] = 1;
                        }
                        else if (A[i*n+j]==0 && neighbour_live_cell==3) {
                                B[i*n+j] = 1;
                        }
                        else {
                                B[i*n+j] = 0;
                        }
                }
        }
}

// Funciones de gpu_simn

__device__ int getat(int *a, int n, int x, int y){
        if (x >= 0 && y >= 0 && x < n && y < n)
                return a[x*n+y];
        return 0;
}

__device__ int count_live_gpu(int n, int *a, int r, int c){
        count += (getat(a,n,r-1,c));
        count += (getat(a,n,r-1,c+1));
        count += (getat(a,n,r,c-1));
        count += (getat(a,n,r+1,c));
        count += (getat(a,n,r+1,c+1));
        return count;
}

__global__ void gpu_sim(int n, int *A, int *B){
        //Se crea el tid de
        int tidx = blockIdx.x * blockDim.x  +  threadIdx.x;
        int tidy = blockIdx.y * blockDim.y  +  threadIdx.y;

        int pos = tidx * n + tidy;

        int neighbour_live_cell = count_live_gpu(n,A,tidx,tidy);
        if (A[pos]==1 && (neighbour_live_cell==2 || neighbour_live_cell==3)) { B[pos] = 1; }
        else if (A[pos]==0 && neighbour_live_cell==3) { B[pos] = 1; }
        else { B[pos] = 0; }

int main(int argc, char **argv){

        if(argc != 8){
                exit(EXIT_FAILURE);
        }

        //Recibir  los parametros
        int gpu_id = atoi(argv[1]);
        int n = atoi(argv[2]);
        int seed = atoi(argv[3]);
        int pasos = atoi(argv[4]);
        int nt = atoi(argv[5]);
        int Bloque = atoi(argv[6]);
        int modo = atoi(argv[7]);
        const char* modes[2] = {"CPU", "GPU"};

        // setear numero de threads OpenMP
        omp_set_num_threads(nt);

        // Crear las matrices en host
        int *A = new int[n*n];
        int *B = new int[n*n];

        crearMatriz(n, A, seed);


        //Imprimir estado incial si n es menor a 128
        if (n<128){
                printf("Estado Inicial:"); fflush(stdout);
                row_line(n);
                printmat (n, A);
        }

        //Alojar matriz
        int *Ad, *Bd;
        cudaMalloc(&Ad, sizeof(int)*n*n);
        cudaMalloc(&Bd, sizeof(int)*n*n);
        cudaMemcpy(Ad, A, sizeof(int)*n*n, cudaMemcpyHostToDevice);
        cudaMemcpy(Bd, B, sizeof(int)*n*n, cudaMemcpyHostToDevice);

        // Se define el bloque y la grilla:
        dim3 block(Bloque, Bloque);
        dim3 grid(n/Bloque, n/Bloque);

        char tecla = getchar();
        printf("Calculando.\n");

        int* actual;
        int* sig;

        // Se comienza la cuenta
        cudaEvent_t start, stop; float msecs;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int i=0; i<pasos; i++){
                switch(modo){
                        case 0:
                                cpu_sim(n, A, B);
                                igualar(n, A, B);
                                break;
                        case 1:
                                if ((i % 2) == 0) { actual = Ad; sig = Bd; }
                                else { actual = Bd; sig = Ad; }
                                gpu_sim<<<grid, block>>>(n, actual, sig);
                }
        }

        //Termina la cuenta
        cudaDeviceSynchronize();        cudaEventRecord(stop);
        cudaEventSynchronize(stop);     cudaEventElapsedTime(&msecs, start, stop);

        // Copiar a host si fue por gpu
        if (modo == 1) {
                cudaMemcpy(B, actual, n*n*sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (n<128){
                printf("Despues de %d iteraciones:", pasos); fflush(stdout);
                row_line(n);
                printmat(n,B);
        }

        cudaFree(Ad);
        cudaFree(Bd);
        free(A);

        printf("Listo");
        tecla = getchar();

        printf("ok: tiempo: %f segs\n", msecs/1000.0f); fflush(stdout);
        printf("ok: tiempo: %f milisegs\n", msecs); fflush(stdout);

        printf ("Listo\n");
        exit(EXIT_SUCCESS);
}