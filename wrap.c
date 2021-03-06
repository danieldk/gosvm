#include <stdlib.h>
#include <string.h>
#include <svm.h>

#include "wrap.h"


svm_node_t *gosvm_nodes_new(size_t n)
{
  svm_node_t *nodes = malloc((n + 1) * sizeof(svm_node_t));

  // Terminator
  nodes[n].index = -1;
  nodes[n].value = 0.0;

  return nodes;
}

void gosvm_nodes_free(svm_node_t *nodes)
{
  free(nodes);
}

void gosvm_nodes_put(svm_node_t *nodes, size_t nodes_idx, int idx,
  double value)
{
  nodes[nodes_idx].index = idx;
  nodes[nodes_idx].value = value;
}

svm_parameter_t *gosvm_parameter_new()
{
  svm_parameter_t *param = malloc(sizeof(svm_parameter_t));
  memset(param, 0, sizeof(svm_parameter_t));
  return param;
}

void gosvm_problem_free(svm_problem_t *problem)
{
  free(problem->x);
  free(problem->y);
  free(problem);
}

void gosvm_problem_add_train_inst(svm_problem_t *problem, svm_node_t *nodes,
  double label)
{
  ++problem->l;
  problem->y = realloc(problem->y, problem->l * sizeof(double));
  problem->y[problem->l - 1] = label;
  problem->x = realloc(problem->x, problem->l * sizeof(svm_node_t *));
  problem->x[problem->l - 1] = nodes;
}

svm_problem_t *gosvm_problem_new()
{
  svm_problem_t *problem = malloc(sizeof(svm_problem_t));

  problem->l = 0;
  problem->y = malloc(0);
  problem->x = malloc(0);

  return problem;
}

int *gosvm_labels_new(int n)
{
  int *labels = malloc(n * sizeof(int));
  memset(labels, 0, n * sizeof(int));
  return labels;
}

double *gosvm_probs_new(svm_model_t *model)
{
  int nClasses = svm_get_nr_class(model);
  double *probs = malloc(nClasses * sizeof(double));
  memset(probs, 0, nClasses * sizeof(double));
  return probs;
}

double gosvm_get_double_idx(double *arr, int idx)
{
  return arr[idx];
}

int gosvm_get_int_idx(int *arr, int idx)
{
  return arr[idx];
}

char const *svm_check_parameter_wrap(svm_problem_t *prob,
    svm_parameter_t *param)
{
  return svm_check_parameter(prob, param);
}

int svm_check_probability_model_wrap(svm_model_t const *model)
{
  return svm_check_probability_model(model);
}

void svm_destroy_param_wrap(svm_parameter_t* param)
{
  return svm_destroy_param(param);
}

void svm_get_labels_wrap(svm_model_t const *model, int *label)
{
  svm_get_labels(model, label);
}

void svm_free_and_destroy_model_wrap(svm_model_t *model)
{
  svm_free_and_destroy_model(&model);
}

int svm_get_nr_class_wrap(svm_model_t const *model)
{
  return svm_get_nr_class(model);
}

svm_model_t *svm_load_model_wrap(char const *filename)
{
  return svm_load_model(filename);
}

svm_model_t *svm_train_wrap(svm_problem_t *prob, svm_parameter_t *param)
{
  return svm_train(prob, param);
}

int svm_save_model_wrap(svm_model_t const *model, char const *filename)
{
  return svm_save_model(filename, model);
}

double svm_predict_probability_wrap(svm_model_t const *model, 
    svm_node_t const *x, double *prob_estimates)
{
  return svm_predict_probability(model, x, prob_estimates);
}

double svm_predict_wrap(svm_model_t const *model, svm_node_t *nodes)
{
  return svm_predict(model, nodes);
}
