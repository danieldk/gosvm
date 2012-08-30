package gosvm

/*
#cgo LDFLAGS: -lsvm
#include "wrap.h"
*/
import "C"

import (
	"runtime"
	"sort"
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

type byIndex struct{ FeatureVector }

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
	// libsvm requires the features to be sorted. So, we sort it, but
	// let's not touch the user's slice.
	features := make(FeatureVector, len(trainInst.Features))
	copy(features, trainInst.Features)
	sort.Sort(byIndex{features})

	nodes := C.nodes_new(C.size_t(len(features)))

	for idx, val := range features {
		C.nodes_put(nodes, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	C.problem_add_train_inst(problem.problem, nodes, C.double(trainInst.Label))
}

// Helper functions

// Interface for sorting of feature vectors by feature index.

func (fv byIndex) Len() int {
	return len(fv.FeatureVector)
}

func (fv byIndex) Swap(i, j int) {
	fv.FeatureVector[i], fv.FeatureVector[j] =
		fv.FeatureVector[j], fv.FeatureVector[i]
}

func (fv byIndex) Less(i, j int) bool {
	return fv.FeatureVector[i].Index < fv.FeatureVector[j].Index
}
