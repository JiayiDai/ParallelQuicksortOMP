/*
	qsortOpenMP.cpp

	Parallel quicksort in OpenMP

	compile in Linux: g++ -o qsort -fopenmp qsortOpenMP.cpp
	run in Linux: ./qsort <num_of_threads> <size_of_array>

	It's a parallel quicksort program with threads of t.
	The idea is:
		1. partition array equally (with chunk_size = size / t) for each thread
		2. do local quicksort in each thread (time complexity is O((n/t) * lg(n/t)))
		3. merge all local sorted parts in a binary tree (with depth lg(t))

*/

#include <iostream>
#include <omp.h>
#include <math.h>
#include <ctime>

void printArray(int* A, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << "A[" << i << "]" << " is " << A[i] << "\n";
	}
	std::cout << "\n";
}

/*
	getRandomArray(array, size)

	fill array with random integers in the range from 0 to RAND_MAX
*/
void getRandomArray(int* array, int size) {
	int i = 0;
	while (i < size) {
		array[i] = rand() % RAND_MAX;
		i++;
	}
}

/*
	partition(array, int, size)

	divide the array into two parts (elements < pivot and elements > pivot)
	obtained from AUCSC 310 lecture slides
*/
int partition(int* array, int start, int end) {
	int pivot = array[end];
	int smallerCount = start;
	for (int j = start; j < end; j++) {
		if (array[j] <= pivot) {
			int temp = array[j];
			array[j] = array[smallerCount];
			array[smallerCount] = temp;
			smallerCount += 1;
		}
	}
	array[end] = array[smallerCount];
	array[smallerCount] = pivot;
	return smallerCount;
}

/*
	quickSort(array, start, end)

	call partition to divide array into two parts - divide
	then recursively call itself on the two parts in two threads - conquer
*/
void quickSort(int* array, int start, int end) {
	if (start < end) {
		int pivotLoc = partition(array, start, end);
		quickSort(array, start, pivotLoc - 1);//left
		quickSort(array, pivotLoc + 1, end);//right
	}
}

/*
	merge(array, merged_array, start, end)

	merge two parts in an array and put the merged one of the
	two parts back to the array in their original space

	assuming two parts have the same size
*/
void merge(int* array, int start, int end) {
	int merged_size = end - start + 1;
	int* merged_array = (int*)malloc(merged_size * sizeof(int));
	int i = start, j = start + merged_size / 2, k = 0;
	int j_start = start + merged_size / 2;
	while (i < j_start && j < end + 1) {//merge
		merged_array[k++] = array[i] < array[j] ? array[i++] : array[j++];
	}
	while (i < j_start) {//fill the rest
		merged_array[k++] = array[i++];
	}
	while (j < end + 1) {
		merged_array[k++] = array[j++];
	}
	for (int i = start; i < end + 1; i++) {
		array[i] = merged_array[i - start];
	}
	free(merged_array);
}

void initializeLevelDone(int* level_done, int size, int thread_count) {
	level_done[0] = thread_count;
	for (int i = 1; i < size; i++) {
		level_done[i] = 0;
	}
}


int main(int argc, char** argv) {

	int size = atoi(argv[2]);
	int thread_count = strtol(argv[1], NULL, 10);
	int chunk_size = size / thread_count;
	//set up initial A
	int* A = (int*)malloc(size * sizeof(int));
	getRandomArray(A, size);
	//printArray(A, size);

	int merge_depth = (int)log2(thread_count);
	int* level_done = (int*)malloc(merge_depth * sizeof(int));
	initializeLevelDone(level_done, merge_depth, thread_count);

	//time counting starts
	double start_time = omp_get_wtime();

	//creating threads
#	pragma omp parallel num_threads(thread_count) 
	{
		int my_rank = omp_get_thread_num();
		int my_index = my_rank;//initial my_index is my_rank
		int my_first = my_rank * chunk_size;
		int my_last = my_first + chunk_size; //excluded

		//local quicksort
		quickSort(A, my_first, my_last - 1);
		
#		pragma omp barrier

		//merge with a binary tree
		int step = 0;
		while (step < merge_depth) {
			int merge_size = chunk_size * (int)pow(2, step + 1);
			int level_merge_count = thread_count/ (int)pow(2, step);//total num of merges on this level
			if (my_index % 2 == 0) {//even index threads do merge
				//consumer and producer issue here
				//when merging two parts that need to be merged first
				//busy waiting
				//when the merges of the previous level of the current one is not done
				while (level_done[step] != level_merge_count);
				merge(A, my_first, my_first + merge_size - 1);
#				pragma omp atomic
				level_done[step + 1]++;
			}
			else {//no work for odd index threads
				break;
			}
			step++;
			my_index = my_index / 2;//index changes for the next layer
		}
	}//threads except thread 0 have done their work

	//time counting ends
	double finish_time = omp_get_wtime();
	double elapsed_time = finish_time - start_time;
	//printArray(A, size);
	std::cout << "with array size = " << size << " and thread_count = " << thread_count << ", elapsed time is " << elapsed_time << "\n";

	return 0;
}