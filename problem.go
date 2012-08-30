package libsvm

/*
#cgo LDFLAGS: -lsvm
#include "wrap.h"
*/
import "C"

import (
	"runtime"
)

type Node struct {
	Index int
	Value float64
}

// XXX - Indices must be sorted in ascending order!
type TrainingVector struct {
	Label float64
	Nodes []Node
}

type Problem struct {
	problem *C.svm_problem_t
}

func NewProblem() *Problem {
	cProblem := C.problem_new()
	problem := &Problem{cProblem}

	runtime.SetFinalizer(problem, func(p *Problem) {
		println("finalizing problem")
		C.problem_free(p.problem)
	})

	return problem
}

func ProblemFromSlice(data [][]float64) *Problem {
	problem := NewProblem()

	for exIdx, vals := range data {
		nodes := make([]Node, len(vals))

		for valIdx, val := range vals {
			nodes[valIdx] = Node{valIdx + 1, val}
		}

		trainVec := TrainingVector{float64(exIdx), nodes}
		problem.AddTrainingVector(trainVec)
	}

	return problem
}

func cNodes(nodes []Node) *C.svm_node_t {
	n := C.nodes_new(C.size_t(len(nodes)))

	for idx, val := range nodes {
		C.nodes_put(n, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	return n
}

func (problem *Problem) AddTrainingVector(trainVec TrainingVector) {
	nodes := C.nodes_new(C.size_t(len(trainVec.Nodes)))

	for idx, val := range trainVec.Nodes {
		C.nodes_put(nodes, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	C.problem_add_trainvec(problem.problem, nodes, C.double(trainVec.Label))
}

func (model *Model) Predict(nodes []Node) float64 {
	cn := cNodes(nodes)
	defer C.nodes_free(cNodes(nodes))
	return float64(C.svm_predict_wrap(model.model, cn))

}
