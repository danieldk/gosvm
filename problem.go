package gosvm

/*
#cgo LDFLAGS: -lsvm
#include "wrap.h"
*/
import "C"

import (
	"runtime"
)

// Represents a feature and its value. The Index of a feature is used
// to uniquely identify the feature, and should start at 1.
type FeatureValue struct {
	Index int
	Value float64
}

// Sparse feature vector, represented as the list (slice) of non-zero
// features.
type FeatureVector []FeatureValue

// Training instance, consisting of the label of the instance and
// its feature vector. In classification, the label is an integer
// indicating the class label. In regression, the label is the
// target value, which can be any real number. The label is not used
// for one-class SVMs.
type TrainingInstance struct {
	Label    float64
	Features FeatureVector
}

// A problem is a set of instances and corresponding labels.
type Problem struct {
	problem *C.svm_problem_t
}

func NewProblem() *Problem {
	cProblem := C.problem_new()
	problem := &Problem{cProblem}

	runtime.SetFinalizer(problem, func(p *Problem) {
		C.problem_free(p.problem)
	})

	return problem
}

// This is a helper function that creates a problem from a two-dimensional
// slice. Consider the following example:
//
//   data := [][]float64{{1.0, 0.0, 1.0}, {-1, 0.0, -1}}
//   problem := svm.ProblemFromSlice(data)
//
// This fragment creates a problem consisting of two instances. For each
// instance, three feature values are specified.
func ProblemFromSlice(data [][]float64) *Problem {
	problem := NewProblem()

	for exIdx, vals := range data {
		nodes := make([]FeatureValue, len(vals))

		for valIdx, val := range vals {
			nodes[valIdx] = FeatureValue{valIdx + 1, val}
		}

		trainVec := TrainingInstance{float64(exIdx), nodes}
		problem.Add(trainVec)
	}

	return problem
}

func cNodes(nodes []FeatureValue) *C.svm_node_t {
	n := C.nodes_new(C.size_t(len(nodes)))

	for idx, val := range nodes {
		C.nodes_put(n, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	return n
}

func (problem *Problem) Add(trainInst TrainingInstance) {
	// BUG(danieldk): Feature indices should be sorted in ascending order,
	// do this when adding a TrainingInstance to a Problem.
	nodes := C.nodes_new(C.size_t(len(trainInst.Features)))

	for idx, val := range trainInst.Features {
		C.nodes_put(nodes, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	C.problem_add_train_inst(problem.problem, nodes, C.double(trainInst.Label))
}
