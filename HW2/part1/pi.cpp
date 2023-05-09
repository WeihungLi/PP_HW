// HW2-part1.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
/*sem_t semaphore; // 旗標
volatile long long int number_in_circle;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;*/
typedef struct storageList{
    int time;
    int index;
    int sum;
}storage;

void* monteCarlo(void* da);

int main(int argc, char** argv)
{
    int Nums = atoi(argv[1]);
    int time = atoi(argv[2])/ Nums;
    long long int number_in_circle = 0;
    pthread_t threads[Nums];
    storage* sample = (storage*)malloc((Nums)*sizeof(int));
    //sem_init(&semaphore, 0, 0);
    for (int i = 0; i < Nums; i++) {
        sample[i].time = time;
        sample[i].index = i;
        sample[i].sum = 0;
        pthread_create(&threads[i], NULL, monteCarlo, (void*)&sample[i]); // 建立子執行緒
    }
    //sem_post(&semaphore);
    for (int i = 0; i < Nums; i++) {
        pthread_join(threads[i], NULL);
    }
    for (int i = 0; i < Nums; i++) {
        number_in_circle+=sample[i].sum;
    }
    double pi = 4 * number_in_circle / (time* (double)Nums);
    std::cout << "The Pi is almost equal "<<pi<< " in " << time << " times simulations\n" << std::endl;
    return 0;
}
void*  monteCarlo(void* da) {
    float max = 1.0;
    float min = -1.0;
    long long int sample = 0;
    storage* data = (storage*)da;
    int number_of_tosses =  data->time;
    unsigned 2/29 = (unsigned)time(NULL); // 取得時間序列
    srand(seed); // 以時間序列當亂數種子
    for (int toss = 0; toss < (number_of_tosses); toss++) {
        double x = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double y = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0)
            sample++;
    }
    data->sum=sample;
    std::cout<< "Index: "<<data->index<<" Sample: "<<data->sum<< "\n" << std::endl;
    /*sem_wait(&semaphore); // 等待工作
    number_in_circle += sample;
    sem_post(&semaphore);
    */

    pthread_exit(NULL);
}
// 執行程式: Ctrl + F5 或 [偵錯] > [啟動但不偵錯] 功能表
// 偵錯程式: F5 或 [偵錯] > [啟動偵錯] 功能表

// 開始使用的提示: 
//   1. 使用 [方案總管] 視窗，新增/管理檔案
//   2. 使用 [Team Explorer] 視窗，連線到原始檔控制
//   3. 使用 [輸出] 視窗，參閱組建輸出與其他訊息
//   4. 使用 [錯誤清單] 視窗，檢視錯誤
//   5. 前往 [專案] > [新增項目]，建立新的程式碼檔案，或是前往 [專案] > [新增現有項目]，將現有程式碼檔案新增至專案
//   6. 之後要再次開啟此專案時，請前往 [檔案] > [開啟] > [專案]，然後選取 .sln 檔案
