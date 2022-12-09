#include "progress.h"
#include "utils.h"

int total_main_seq_runs = 1;
int current_iteration_count = 0;

void init_progress() {
    total_main_seq_runs = 1;
    current_iteration_count = 0;
}

void print_total_progress() {
    const int total_iteration_count_per_seq = DATASET_NUMBER * (REPEAT_COUNT_REALLOC + REPEAT_COUNT_ONLY_PARALLEL
    + ( (REPEAT_COUNT_ONLY_PARALLEL == 0) ? 0 : REPEAT_COUNT_ONLY_PARALLEL_WARMUP_COUNT) );

    /*logs( "total_iteration_count_per_seq(" + std::to_string(total_iteration_count_per_seq) + ")"
    + " " +  );*/
    // dataset number * (number of iterations total, with warmup time if applicable)
    int total_iteration_count = total_iteration_count_per_seq * total_main_seq_runs;
    int progress = 100 * double(current_iteration_count) / double(total_iteration_count);
    logs( std::to_string(progress) + "% ");
}
