#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    double* score_old;
    double* score_new;
    score_old = (double*)malloc(sizeof(double) * numNodes);
    if (!score_old) {
        free(score_old);
        return;
    }
    score_new = (double*)malloc(sizeof(double) * numNodes);
    if (!score_new) {
        free(score_new);
        return;
    }

    for (size_t i = 0; i < numNodes; i++)
    {
        score_old[i] = equal_prob;

    }

    // initialize variables required for page_rank algorithm
    double global_diff;
    bool converged = false;
    double no_leave_prob;
    // Algorithm part
    while (!converged) {
        // iterate over all nodes in graph
        no_leave_prob = 0.0;
        #pragma omp parallel for reduction(+:no_leave_prob) schedule(dynamic, 16)
        for (size_t i = 0; i < numNodes; i++) {
            // score_new[vi] = sum over all nodes vj reachable from incoming edges
            // { score_old[vj] / number of edges leaving vj  }
            score_new[i] = 0.0;
            if (outgoing_size(g, i) == 0) {
                no_leave_prob += score_old[i];
            }

            const Vertex* income_arr = incoming_begin(g, i);
            int num_incoming = incoming_size(g, i);
            for (size_t j = 0; j < num_incoming; j++) {
                score_new[i] += score_old[income_arr[j]] / outgoing_size(g, income_arr[j]);
            }

            // score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
            score_new[i] = damping * score_new[i] + (1.0 - damping) / numNodes;
        }

        // compute how much per-node scores have changed
        // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
        global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (size_t i = 0; i < numNodes; i++) {
            // score_new[vi] += sum over all nodes v in graph with no outgoing edges                
            score_new[i] += damping * no_leave_prob * equal_prob;
            global_diff += std::abs(score_new[i] - score_old[i]);
        }

        // update score_old and check convergence
        std::swap(score_new, score_old);
        converged = (global_diff < convergence);
    }

    // final result
    for (size_t i; i < numNodes; i++) {
        solution[i] = score_old[i];
    }
    free(score_old);
    free(score_new);
}
