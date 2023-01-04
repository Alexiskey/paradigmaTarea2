#include <cstdio>
#include <cstdlib>

#include <stdlib.h>
#include <stdio.h>

#include <random>
#include <omp.h>
#include <cuda.h>

//#define BSIZE2D 32

// Imprimos la tabla de juego vacia	n y m las filas y columnas
void tabla(int n,int *m,const char* msg){
	printf("%s\n", msg);
 	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			printf("%i ", m[i*n + j]);
        	}
        	printf("\n");
        }
}
// Rellenamos la tabla con valores aleatorios	entre "o" y "" 
void rellenar(int n, int *m, int seed){
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
		for(int h=start; h<n && h<end; ++h){
			m[k*n+h]=dist(mt);
		}
        }
    	}
}
// Contamos la vecindad
int contadorCPU(int *m, int n, int a, int b){
	//int i 	  = 0;
	//int j	  = 0;
	int count = 0;
	#pragma omp parallel for
    	for(int i=a-1; i<=a+1; ++i){
        	for(int j=b-1; j<=b+1; ++j){
            		// fila x columna
            		//float sum=0.0f;
            		if ((i==a && j==b) || (i<0 || j<0) || (i>=n || j >=n)){
                                continue;
                        }
                        if (m[i*n+j] == 1){
                                count++;
                        }
            	//c[i*n+j] = sum;
        	}
    	}
    	return count;
}
// Contamos vencidad en GPU
__device__ int contadorGPU(int *m, int n, int a, int b){
	//int i 	  = 0;
	//int j	  = 0;
	int count = 0;
	//#pragma omp parallel for
    	for(int i=a-1; i<=a+1; ++i){
        	for(int j=b-1; j<=b+1; ++j){
            		// fila x columna
            		//float sum=0.0f;
            		if ((i==a && j==b) || (i<0 || j<0) || (i>=n || j >=n)){
                                continue;
                        }
                        if (m[i*n+j] == 1){
                                count++;
                        }
            	//c[i*n+j] = sum;
        	}
    	}
    	return count;
}
// Simulacion de AC con CPU
void cpu_sim(int *m, int *b, int n){
	int contador;
	#pragma omp parallel for
	for (int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			contador = contadorCPU(m, n, i, j);
			if (m[i*n+j]==1 && (contador==2 || contador==3)){ 
				b[i*n+j]=1;
			}
			else if(m[i*n+j]==0 && contador==3){
				b[i*n+j]=1;
			}
			else{
				b[i*n+j]=0;
			}	
		}
	}
}
// Simulacion de AC con GPU
__global__ void gpu_sim(int n, int *a, int *b){

	int tidx = blockIdx.x * blockDim.x  +  threadIdx.x;
	int tidy = blockIdx.y * blockDim.y  +  threadIdx.y;
	int contador;
	// IMPORTANTE: filtrar los threads que estan fuera del dominio
    	if(tidx < n && tidy < n){
    		contador = contadorGPU(a, n, tidy, tidx);
    		if (a[tidy*n+tidx]==1 && (contador==2 || contador==3)){
    			b[tidy*n+tidx]=1;
    		}
    		else if (a[tidy*n+tidx]==0 && contador==3){
			b[tidy*n+tidx]=1;
		}
		else{
			b[tidy*n+tidx]=0;
		}
	}
}
// Main del juego
int main(int argc, char **argv){
	if(argc !=8){
		fprintf(stderr, "corre como ./prog \ngpu-id // n // seed // pasos // nthreads // nbloques // mode\nmode: normal | mem-comp | cpu-OMP\n\n");
		exit(EXIT_FAILURE);
	}
	int gpuid = atoi(argv[1]); //idgpu
	int n     = atoi(argv[2]); //tamaño matriz nxn
	int seed  = atoi(argv[3]); //seed
	int pasos = atoi(argv[4]); //pasos
	 
	int nt   = atoi(argv[5]); //n° de threads
	int nb   = atoi(argv[6]); //bloques
	int mode = atoi(argv[7]); //modo
	//const char* modes[3] = {"normal", "mem-comp", "cpu-OMP"}; 
	
	int *a = new int[n*n];
	int *b = new int[n*n];
	int *ad;
	int *bd;
	
	float msecs = 0.0f;
	
	if (mode 	== 0){
		printf("CPU: %d matriz || %d pasos || %d threads.\n",n ,pasos ,nt);
	}
	else {
		printf("GPU %d threads %d bloques.\n", nt, nb);
		cudaSetDevice (gpuid);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpuid);
		printf("GPU en uso: %s\n", prop.name);
	}	

	omp_set_num_threads(nt);
	rellenar(n,a,seed);
	
	if (n<128){
		tabla(n,a,"_____Tabla inicial______");
		printf("Presiona Enter para seguir");
		char tecla = getchar();
		printf("Calculando................CON %d PASOS\n",pasos);
	}
	
	cudaMalloc(&ad, sizeof(int)*n*n);
	cudaMalloc(&bd, sizeof(int)*n*n);
	cudaMemcpy(ad, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);
	
	dim3 block(nb, nb);
	dim3 grid((n+nb-1)/nb, (n+nb-1)/nb, 1);
	//char tecla = getchar();
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	if(mode == 0){
		for(int i=0; i<pasos; ++i){
			cpu_sim(a, b, n);
			std::swap(a,b);
		}
	}
	else{
		for(int i=0; i<pasos; ++i){
			gpu_sim<<<grid, block>>>(n, ad, bd);
			std::swap(ad,bd);
		}
	}
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecs, start, stop);
	
	if (mode == 1) {
		cudaMemcpy(b,bd,sizeof(int)*n*n, cudaMemcpyDeviceToHost);
	}
	
	if (n<128){
		tabla(n,b,"_____Tabla final_____");
	}
	
	cudaFree(ad);
	cudaFree(bd);
	free(a);
	
	//printf("Ultima ejecucion (Enter)");
	//char tecla = getchar();
	
	if(mode == 0){
		printf("Speedout en CPU: %f\n",msecs/1000.0f);
	}
	else{
		printf("Speedout en GPU: %f\n",msecs/1000.0f);
	}
	//printf("tiempo : %f sec\n", msecs/1000.0f);
	
	printf ("Terminado\n");
	//exit(EXIT_SUCCESS);
}

