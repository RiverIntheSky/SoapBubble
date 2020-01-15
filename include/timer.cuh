# pragma once

# include "../include/header.cuh"

class Timer
{
private:
    cudaEvent_t start;
    cudaEvent_t stop;
    float timeElapsed;
public:
    Timer();
    ~Timer();

    void startTimer();
    float stopTimer();
};
