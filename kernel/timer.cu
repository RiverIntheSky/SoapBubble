# include "../include/timer.cuh"

Timer::Timer() : timeElapsed(0.0f)
{
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
}
Timer::~Timer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
void Timer::startTimer()
{
    cudaEventRecord(start, 0);
}
float Timer::stopTimer()
{
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkCudaErrors(cudaEventElapsedTime(&timeElapsed, start, stop));
    return timeElapsed;
}